"""
Step 1: Collect student face images using webcam + MediaPipe for detection.
Usage: python collect_faces.py --name "Student_Name" --count 50
"""

import cv2
import mediapipe as mp
import os
import argparse
import time

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Capture face images for a student")
parser.add_argument("--name",  required=True, help="Student name (e.g. 'Sneha_Reddy')")
parser.add_argument("--count", type=int, default=60, help="Number of images to capture")
args = parser.parse_args()

SAVE_DIR = os.path.join("train_images", args.name)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MediaPipe Face Detection ──────────────────────────────────────────────────
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
captured = 0
last_capture_time = 0
CAPTURE_INTERVAL = 0.3  # seconds between captures

print(f"[INFO] Capturing {args.count} images for '{args.name}'. Press 'q' to quit.")

while captured < args.count:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    display = frame.copy()

    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(display, detection)

            now = time.time()
            if now - last_capture_time >= CAPTURE_INTERVAL:
                # Crop face region
                bboxC = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = max(0, int(bboxC.xmin * w) - 20)
                y1 = max(0, int(bboxC.ymin * h) - 20)
                x2 = min(w, int((bboxC.xmin + bboxC.width)  * w) + 20)
                y2 = min(h, int((bboxC.ymin + bboxC.height) * h) + 20)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    filename = os.path.join(SAVE_DIR, f"{captured:04d}.jpg")
                    cv2.imwrite(filename, face_crop)
                    captured += 1
                    last_capture_time = now
                    print(f"  [{captured}/{args.count}] Saved {filename}")

    # HUD
    cv2.putText(display, f"Captured: {captured}/{args.count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display, f"Student: {args.name}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Face Collection — press Q to quit", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[DONE] Collected {captured} images in '{SAVE_DIR}'")