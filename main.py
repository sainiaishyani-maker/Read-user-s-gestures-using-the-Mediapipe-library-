import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    return math.hypot(point1.x - point2.x, point1.y - point2.y)

def detect_jutsu(multi_hand_landmarks):
    """
    Analyzes hand landmarks to detect specific Naruto hand signs.
    Currently configured to detect a simplified 'Tiger Seal' (Tora).
    """
    # Naruto signs usually require both hands
    if len(multi_hand_landmarks) != 2:
        return None
    
    hand1 = multi_hand_landmarks[0].landmark
    hand2 = multi_hand_landmarks[1].landmark
    
    # Extract Index Finger Tips (Landmark 8 in MediaPipe)
    index_tip1 = hand1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_tip2 = hand2[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate distance between the two index finger tips
    distance = calculate_distance(index_tip1, index_tip2)
    
    # If the index fingers are touching/crossed (distance is very small)
    if distance < 0.05:
        return "Jutsu: TIGER SEAL (Tora)!"
    
    return None

def main():
    # Start capturing video from the default webcam (0)
    cap = cv2.VideoCapture(0)
    
    # Setup MediaPipe Hands model
    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    ) as hands:
        
        print("Starting camera... Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Flip frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB (MediaPipe requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = hands.process(rgb_frame)
            
            # If hands are detected
            if results.multi_hand_landmarks:
                # Draw the hand skeleton on the frame
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                
                # Check for hand signs
                detected_jutsu = detect_jutsu(results.multi_hand_landmarks)
                
                # Display the detected jutsu on the screen
                if detected_jutsu:
                    cv2.putText(frame, detected_jutsu, (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4)
                    
            # Show the final image
            cv2.imshow('Naruto Hand Sign Detector', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()