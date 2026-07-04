import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import json
import time
import os

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

def extract_fingerprint(hand_landmarks_list):
    # UPGRADE: Now accepts 1 or 2 hands!
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

# --- 1. LOAD EXISTING TEMPLATES ---
templates = {}
if os.path.exists("jutsu_templates.json"):
    with open("jutsu_templates.json", "r") as f:
        templates = json.load(f)
    print(f"Loaded existing file with {len(templates)} saved signs.")

# ================================================================
# We only want to learn the new POINTER sign right now
SIGNS_TO_LEARN = ["dog"] 
FRAMES_PER_SIGN = 3  
# ================================================================

capture_queue = [(sign, i + 1) for sign in SIGNS_TO_LEARN for i in range(FRAMES_PER_SIGN)]
current_idx = 0
cap = cv2.VideoCapture(0)

print(f"\n--- 1-HAND CAPTURE STUDIO ---")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    current_time = time.time()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    detection_result = detector.detect_for_video(mp_image, int(current_time * 1000))
    hands_count = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                cx_draw, cy_draw = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx_draw, cy_draw), 3, (0, 255, 255), -1) 

    if current_idx < len(capture_queue):
        target_sign, frame_num = capture_queue[current_idx]
        cv2.putText(frame, f"RECAPTURE: {target_sign.upper()} ({frame_num}/{FRAMES_PER_SIGN})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Hands detected: {hands_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hands_count > 0 else (0,0,255), 2)
    else:
        cv2.putText(frame, "ALL DONE! Press 'Q' to quit.", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capture Studio", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # UPGRADE: Allows saving as long as AT LEAST 1 hand is visible
    if key == 32 and current_idx < len(capture_queue):
        if hands_count > 0:
            fingerprint = extract_fingerprint(detection_result.hand_landmarks)
            if fingerprint:
                if frame_num == 1: templates[target_sign] = []
                templates[target_sign].append(fingerprint)
                print(f"Saved {target_sign.upper()} frame {frame_num}/{FRAMES_PER_SIGN}!")
                current_idx += 1
                with open("jutsu_templates.json", "w") as f:
                    json.dump(templates, f)
            
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()