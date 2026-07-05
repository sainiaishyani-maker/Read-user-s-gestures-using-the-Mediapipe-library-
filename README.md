<h1 align="center">🥷 Real-Time Jutsu VFX Engine</h1>

<p align="center">
  <b>A computer vision engine that detects complex hand-sign combos to trigger live visual effects.</b>
</p>

---

## 📖 About The Project

The Real-Time Jutsu VFX Engine is an interactive computer vision application that bridges the gap between physical hand gestures and digital visual effects. By utilizing advanced AI tracking, the project allows users to cast elemental "Jutsus" in real-time simply by performing specific hand-sign sequences in front of a webcam.

The application moves beyond basic gesture recognition by implementing a sophisticated, data-driven template matching system. Rather than relying on rigid, pre-programmed math for every finger joint, the engine treats hand configurations as dynamic mathematical fingerprints, allowing for a more fluid and responsive user experience.

## Explanation

The engine operates through a three-stage pipeline designed for both performance and accuracy:

1.  **Gesture Acquisition (The Data Layer):** 
    Using `capture_templates.py`, the system maps the 21 3D landmarks provided by Google MediaPipe. It normalizes these coordinates to create a unique mathematical signature for every hand sign, which is then stored in `jutsu_templates.json`. This approach allows for "Multi-Frame Variance," meaning the engine can recognize the same sign even if the hand is tilted or at a slight distance.

2.  **Sequence Logic (The Inference Layer):** 
    The core `main.py` script doesn't just look for static signs; it maintains a state-machine that tracks sequences. It requires specific signs to be performed in a precise order (e.g., *Snake* ➔ *Ram* ➔ *Tiger*) within a set time window, effectively filtering out accidental movements and false positives.

3.  **VFX Rendering (The Output Layer):** 
    Once a valid sequence is confirmed, the engine triggers an animation. By using `convert.py`, the system pre-processes MP4 assets into luminance-masked sprites. During runtime, it overlays these transparent animations onto the live camera feed, creating a seamless, augmented-reality effect.

## 📂 Project Structure

```text
├── main.py                  # Core engine: handles webcam feed, combo tracking, and VFX rendering
├── capture_templates.py     # Studio tool to record new gestures and save them to the dataset
├── convert.py               # Scripts to strip black backgrounds from MP4s and generate transparent PNGs
├── jutsu_templates.json     # The JSON database storing the mathematical fingerprints of all hand signs
├── fire_sprites/            # Directory containing extracted transparent fire VFX frames
├── wind_sprites/            # Directory containing extracted transparent wind VFX frames
└── sphere_sprites/          # Directory containing extracted transparent water sphere VFX frames
```
## 📊 Datasets
​
This project relies on a custom, dynamically generated dataset rather than pre-trained image classification models.
​jutsu_templates.json: This file acts as the project's brain. When a user creates a new sign using capture_templates.py, the system calculates the relative distances between all 21 hand landmarks provided by MediaPipe.
​Multi-Frame Variance: To ensure bulletproof accuracy, the dataset stores multiple angles and micro-variations of each sign. During live tracking, the engine compares the user's current hand coordinates against this dataset using a strict threshold to determine if a valid Jutsu sign is being cast.
​
## ⚙️ Installation
​1. Clone the repository
   
```git clone [https://github.com/sainiaishyani-maker/Read-user-s-gestures-using-the-Mediapipe-library-.git](https://github.com/sainiaishyani-maker/Read-user-s-gestures-using-the-Mediapipe-library-.git)
cd Read-user-s-gestures-using-the-Mediapipe-library-
```
3. Set up a virtual environment (Optional but recommended)
```
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
4. Install required dependencies
```
pip install opencv-python mediapipe numpy
```
5. Run the engine
```
python main.py
```
## 🙏 Acknowledgements

* **[Google MediaPipe](https://developers.google.com/mediapipe):** For providing the incredibly fast and robust 3D hand-tracking machine learning pipeline that serves as the foundation for this project.
* **[OpenCV](https://opencv.org/):** For handling the real-time matrix operations, webcam feed rendering, and essential alpha-blending logic.
* **Community Resources:** Special thanks to the open-source community for the VFX source assets and the documentation that helped in implementing the luminance-masking techniques.
---

<p align="center">
  <b>Thank you for checking out the Jutsu VFX Engine!</b>
</p>
