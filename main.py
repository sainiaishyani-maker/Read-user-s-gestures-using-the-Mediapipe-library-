import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize the MediaPipe Tasks Detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

# Start Webcam Feed
cap = cv2.VideoCapture(0)
print("Hand Tracking Active! Press 'q' to quit.")

# Create a custom, manually increasing timestamp fallback
frame_count = 0

# Initialize the detector session
with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # FIX: Generate a reliable, strictly increasing timestamp using frame count
        # Assuming roughly 30 frames per second, we increment by ~33ms per frame
        timestamp_ms = int(frame_count * 33.33)
        frame_count += 1
        
        # Detect hands!
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Draw the tracked points if hands are found
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for id, landmark in enumerate(hand_landmarks):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Draw visual indicator dots on knuckles/fingertips
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, str(id), (cx + 5, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv2.imshow("Hand Tracking (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()