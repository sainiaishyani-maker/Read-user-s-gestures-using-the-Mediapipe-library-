import cv2
import os
import numpy as np

video_path = "sphere_input.mp4"
output_folder = "sphere_sprites"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
frame_count = 0

print(f"--- EXTRACTING SMOOTH GLOW SPRITES FROM {video_path} ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Convert to a 4-channel image (Blue, Green, Red, Alpha/Transparency)
    bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    # 2. Get the grayscale/brightness version of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. VFX TRICK: Use the brightness directly as the transparency mask!
    # We tweak the contrast slightly (alpha=1.2, beta=-15) so the darks are 
    # truly invisible but the glowing center stays bright and powerful.
    alpha_mask = cv2.convertScaleAbs(gray, alpha=1.2, beta=-15)
    
    # 4. Apply the smooth mask to the image
    bgra_frame[:, :, 3] = alpha_mask
    
    # 5. Save the transparent PNG (this will automatically overwrite your old bad ones)
    filename = os.path.join(output_folder, f"sprite_{frame_count:04d}.png")
    cv2.imwrite(filename, bgra_frame)
    frame_count += 1

cap.release()
print(f"Success! Extracted {frame_count} incredibly smooth frames into '{output_folder}'.")