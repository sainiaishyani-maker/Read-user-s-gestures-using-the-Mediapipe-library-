import cv2
import os

output_folder = "fire_sprites"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture("fire_input.mp4")
frame_count = 0
saved_count = 0

# CHANGE THIS: 3 means save every 3rd frame, 5 means save every 5th frame.
# Setting it to 5 will cut 786 frames down to around 150 frames!
FRAME_SKIP = 5 

print("Processing video frames with optimization... Please wait.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Only process and save the frame if it matches our skip interval
    if frame_count % FRAME_SKIP == 0:
        frame = cv2.resize(frame, (250, 250))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        b, g, r = cv2.split(frame)
        bgra_frame = cv2.merge([b, g, r, alpha_mask])
        
        # Save with a sequential index
        cv2.imwrite(os.path.join(output_folder, f"fire_{saved_count:04d}.png"), bgra_frame)
        saved_count += 1
        
    frame_count += 1

cap.release()
print(f"Done! Successfully reduced and saved {saved_count} optimized frames inside '{output_folder}'.")