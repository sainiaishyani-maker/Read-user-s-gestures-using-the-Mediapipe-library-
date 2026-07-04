import cv2
import os
import numpy as np

def extract_sprites(video_path, output_folder, mode="smooth", frame_skip=1, resize_dim=None):
    """
    mode: "smooth" for alpha-channel glow, "hard" for binary transparency
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    print(f"--- Processing {video_path} into {output_folder} (Mode: {mode}) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_skip == 0:
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)

            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if mode == "hard":
                _, alpha = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            else: # Smooth mode
                alpha = cv2.convertScaleAbs(gray, alpha=1.2, beta=-15)

            bgra[:, :, 3] = alpha
            cv2.imwrite(os.path.join(output_folder, f"sprite_{saved_count:04d}.png"), bgra)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames.")

# --- RUN CONFIGURATION ---
if __name__ == "__main__":
    # Fire uses Hard Thresholding and resizing for efficiency
    extract_sprites("fire_input.mp4", "fire_sprites", mode="hard", frame_skip=5, resize_dim=(250, 250))
    
    # Wind and Sphere use Smooth Alpha extraction
    extract_sprites("wind_input.mp4", "wind_sprites", mode="smooth")
    extract_sprites("sphere_input.mp4", "sphere_sprites", mode="smooth")