import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import os
import json

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# --- LOAD TEMPLATES ---
templates = {}
try:
    with open("jutsu_templates.json", "r") as f:
        templates = json.load(f)
    print(f"Loaded templates for {len(templates)} unique signs successfully!")
except FileNotFoundError:
    print("WARNING: jutsu_templates.json not found!")

# --- SPRITE LOADER ---
fire_folder, wind_folder, sphere_folder = "fire_sprites", "wind_sprites", "sphere_sprites"
fire_images, wind_images, sphere_images = [], [], []

if os.path.exists(fire_folder): fire_images = [cv2.imread(os.path.join(fire_folder, f), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir(fire_folder)) if f.endswith(".png")]
if os.path.exists(wind_folder): wind_images = [cv2.imread(os.path.join(wind_folder, f), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir(wind_folder)) if f.endswith(".png")]
if os.path.exists(sphere_folder): sphere_images = [cv2.imread(os.path.join(sphere_folder, f), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir(sphere_folder)) if f.endswith(".png")]

def overlay_fullscreen_png(background, overlay):
    bh, bw, _ = background.shape
    overlay_resized = cv2.resize(overlay, (bw, bh))
    sprite_bgr = overlay_resized[:, :, :3]
    mask = overlay_resized[:, :, 3] / 255.0
    for c in range(3):
        background[:, :, c] = (mask * sprite_bgr[:, :, c] + (1.0 - mask) * background[:, :, c]).astype('uint8')
    return background

# --- AI TEMPLATE MATCHING ENGINE ---
def extract_fingerprint(hand_landmarks_list):
    # Safely handles 1 or 2 hands now!
    if not hand_landmarks_list: return None
    points = [[lm.x, lm.y, lm.z] for hand in hand_landmarks_list for lm in hand]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    translated = [[p[0]-cx, p[1]-cy, p[2]-cz] for p in points]
    max_dist = max(math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in translated)
    if max_dist == 0: max_dist = 1.0
    fingerprint = []
    for p in translated:
        fingerprint.extend([p[0]/max_dist, p[1]/max_dist, p[2]/max_dist])
    return fingerprint

def match_fingerprint(live_fingerprint):
    if not live_fingerprint or not templates: return None
    MATCH_THRESHOLD = 1.25  
    best_sign, best_score = None, float('inf')
    
    for sign_name, saved_data in templates.items():
        fingerprints_to_check = [saved_data] if isinstance(saved_data[0], float) else saved_data
        for saved_fingerprint in fingerprints_to_check:
            # CRITICAL FIX: Only compares 1-hand poses to 1-hand poses to prevent crashing
            if len(live_fingerprint) != len(saved_fingerprint):
                continue
                
            distance = math.sqrt(sum((a - b)**2 for a, b in zip(live_fingerprint, saved_fingerprint)))
            if distance < best_score:
                best_score = distance
                best_sign = sign_name
            
    if best_score < MATCH_THRESHOLD:
        return best_sign
    return None

# --- STATE VARIABLES ---
active_combo = None 
sign_history = []  
fx_active = None    
fx_start_time = 0.0
fire_frame_index = 0
wind_frame_index = 0
sphere_frame_index = 0

last_logged_sign = None
last_detected_sign = None
sign_held_start = 0.0
STABILIZATION_DELAY = 0.40  

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    detection_result = detector.detect_for_video(mp_image, int(current_time * 1000))
    fx_rendered_this_frame = False
    detected_sign = None  
    hands_visible = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0

    # Timer only applies to Fire and Wind. Sphere stays forever!
    if fx_active and fx_active != "sphere" and (current_time - fx_start_time >= 10.0):
        fx_active, active_combo, sign_history, last_logged_sign, last_detected_sign = None, None, [], None, None

    # --- THE CORE LOGIC ---
    if detection_result.hand_landmarks:
        # Note: We removed 'and not fx_active' here so you can interrupt the Sphere!
        live_fingerprint = extract_fingerprint(detection_result.hand_landmarks)
        detected_sign = match_fingerprint(live_fingerprint)

        if detected_sign:
            detected_sign = detected_sign.lower()
            if detected_sign != last_detected_sign:
                last_detected_sign = detected_sign
                sign_held_start = current_time
            else:
                if current_time - sign_held_start >= STABILIZATION_DELAY:
                    if detected_sign != last_logged_sign:
                        step_was_valid = False
                        
                        # --- STARTERS & INSTANT CASTS ---
                        if detected_sign == "pointer":
                            active_combo, sign_history = None, ["pointer"]
                            fx_active = "sphere"  # Sphere triggers instantly!
                            step_was_valid = True
                        elif detected_sign == "snake": 
                            active_combo, sign_history = "fire", ["snake"]
                            fx_active = None  # Cancels Sphere if it was running
                            step_was_valid = True
                        elif detected_sign == "dog": 
                            active_combo, sign_history = "wind", ["dog"]
                            fx_active = None  # Cancels Sphere if it was running
                            step_was_valid = True
                            
                        # --- COMBOS ---
                        elif active_combo == "fire":
                            if detected_sign == "ram" and len(sign_history) == 1: 
                                sign_history.append("ram"); step_was_valid = True
                            elif detected_sign == "monkey" and len(sign_history) == 2: 
                                sign_history.append("monkey"); step_was_valid = True
                            elif detected_sign == "boar" and len(sign_history) == 3: 
                                sign_history.append("boar"); step_was_valid = True
                            elif detected_sign == "horse" and len(sign_history) == 4: 
                                sign_history.append("horse"); step_was_valid = True
                            elif detected_sign == "tiger" and len(sign_history) == 5:
                                sign_history.append("tiger")
                                fx_active, fx_start_time, active_combo = "fire", current_time, None
                                step_was_valid = True
                                
                        elif active_combo == "wind":
                            if detected_sign == "horse" and len(sign_history) == 1: 
                                sign_history.append("horse"); step_was_valid = True
                            elif detected_sign == "bird" and len(sign_history) == 2:
                                sign_history.append("bird")
                                fx_active, fx_start_time, active_combo = "wind", current_time, None
                                step_was_valid = True

                        if step_was_valid:
                            last_logged_sign = detected_sign
        else:
            last_detected_sign = None

    # --- FX RENDERING ---
    if fx_active == "fire" and fire_images:
        frame = overlay_fullscreen_png(frame, fire_images[fire_frame_index])
        fire_frame_index = (fire_frame_index + 1) % len(fire_images)
        fx_rendered_this_frame = True
    elif fx_active == "wind" and wind_images:
        frame = overlay_fullscreen_png(frame, wind_images[wind_frame_index])
        wind_frame_index = (wind_frame_index + 1) % len(wind_images)
        fx_rendered_this_frame = True
    elif fx_active == "sphere" and sphere_images:
        frame = overlay_fullscreen_png(frame, sphere_images[sphere_frame_index])
        sphere_frame_index = (sphere_frame_index + 1) % len(sphere_images)
        fx_rendered_this_frame = True

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  
        sign_history, fx_active, active_combo, last_logged_sign, last_detected_sign = [], None, None, None, None

    # --- HUD ---
    if fx_rendered_this_frame:
        if fx_active == "sphere":
            cv2.putText(frame, "WATER PRISON SPHERE ACTIVE!!", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,100,100), 3)
            cv2.putText(frame, "Cast another Jutsu or press SPACE to cancel", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            remaining = max(0.0, 10.0 - (time.time() - fx_start_time))
            name, color = ("KATON: GOUKAKYUU!!", (0,0,255)) if fx_active == "fire" else ("FUTON: DAITOPPA!!", (255,255,0))
            cv2.putText(frame, name, (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 3)
            cv2.putText(frame, f"Time: {remaining:.1f}s", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    else:
        status, color = "Waiting for Jutsu...", (255, 255, 255)
        if active_combo == "fire":
            idx = len(sign_history)
            if idx == 1: status, color = "SNAKE OK -> Waiting: RAM [1/6]", (255, 255, 0)
            elif idx == 2: status, color = "RAM OK -> Waiting: MONKEY [2/6]", (0, 255, 255)
            elif idx == 3: status, color = "MONKEY OK -> Waiting: BOAR [3/6]", (0, 165, 255)
            elif idx == 4: status, color = "BOAR OK -> Waiting: HORSE [4/6]", (0, 69, 255)
            elif idx == 5: status, color = "HORSE OK -> FORM TIGER! [5/6]", (0, 0, 255)
        elif active_combo == "wind":
            if len(sign_history) == 1: status, color = "DOG OK -> Waiting: HORSE [1/3]", (255, 255, 0)
            else: status, color = "HORSE OK -> FORM BIRD! [2/3]", (255, 255, 0)
            
        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        history_text = " -> ".join([s.upper() for s in sign_history])
        cv2.putText(frame, f"Logged: [{history_text}]", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        diag_sign = detected_sign.upper() if detected_sign else "NONE"
        box_color = (0, 255, 0) if (detected_sign == last_logged_sign) else (0, 165, 255)
        if not detected_sign: box_color = (100, 100, 100)
        cv2.putText(frame, f"Detected: {diag_sign} ({hands_visible} Hand/s)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    cv2.imshow("Data-Driven Jutsu Engine", frame)
    if key == ord('q'): break

cap.release()
detector.close()
cv2.destroyAllWindows()