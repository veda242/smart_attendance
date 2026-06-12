"""
recognition.py — Core recognition engine.
Handles: MediaPipe detection → FaceNet embedding → Cosine similarity → Time-based threshold
"""

import os
import pickle
import time
import numpy as np
import cv2
import mediapipe as mp

# Disable torch.dynamo to avoid import hangs
os.environ['TORCH_COMPILE'] = 'False'
os.environ['DISABLE_TELEMETRY'] = 'True'

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_FILE         = "embeddings/embeddings.pkl"
COSINE_THRESHOLD   = 0.55    # below this → unknown  (lower = stricter)
TIME_THRESHOLD_SEC = 5.0     # seconds face must be seen before marking present
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


class AttendanceEngine:
    def __init__(self):
        print(f"[ENGINE] Loading models on {DEVICE} ...")

        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.55
        )

        # FaceNet (MTCNN for alignment + InceptionResnetV1 for embedding)
        self.mtcnn   = MTCNN(image_size=160, margin=20, device=DEVICE, keep_all=False)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

        # Load stored embeddings
        with open(EMBED_FILE, "rb") as f:
            data = pickle.load(f)
        self.db_embeddings = data["embeddings"]   # shape (N, 512)
        self.db_labels     = data["labels"]
        print(f"[ENGINE] Loaded {len(self.db_labels)} embeddings | "
              f"Students: {list(set(self.db_labels))}")

        # Time tracker: {name: first_seen_timestamp}
        self.first_seen: dict[str, float] = {}
        # Already marked present this session
        self.marked: set[str] = set()

    # ── Embedding ─────────────────────────────────────────────────────────────
    def _embed(self, face_rgb: np.ndarray):
        """Convert a cropped face (numpy RGB) to normalised 512-d embedding."""
        img = Image.fromarray(face_rgb)
        tensor = self.mtcnn(img)
        if tensor is None:
            return None
        tensor = tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.facenet(tensor).cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    # ── Cosine similarity ─────────────────────────────────────────────────────
    def _identify(self, emb: np.ndarray):
        """
        Compare embedding against DB using cosine similarity.
        Returns (name, similarity) or ("Unknown", score).
        """
        sims = self.db_embeddings @ emb           # dot product of L2-normalised = cosine
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])
        best_name = self.db_labels[best_idx]

        if best_sim >= COSINE_THRESHOLD:
            return best_name, best_sim
        return "Unknown", best_sim

    # ── Time-based threshold ──────────────────────────────────────────────────
    def _update_timer(self, name: str):
        """
        Returns True (and marks present) if name has been continuously
        detected for >= TIME_THRESHOLD_SEC.
        """
        if name == "Unknown" or name in self.marked:
            return False

        now = time.time()
        if name not in self.first_seen:
            self.first_seen[name] = now
            return False

        elapsed = now - self.first_seen[name]
        if elapsed >= TIME_THRESHOLD_SEC:
            self.marked.add(name)
            return True
        return False

    def reset_timer(self, name: str):
        """Call when face disappears from frame — resets accumulation."""
        self.first_seen.pop(name, None)

    # ── Per-frame processing ──────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Process one BGR webcam frame.
        Returns:
            annotated_frame  — frame with bounding boxes + labels
            newly_marked     — list of names marked present this call
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        h, w = frame.shape[:2]
        annotated = frame.copy()
        newly_marked = []
        detected_names = []

        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w) - 15)
                y1 = max(0, int(bb.ymin * h) - 15)
                x2 = min(w, int((bb.xmin + bb.width)  * w) + 15)
                y2 = min(h, int((bb.ymin + bb.height) * h) + 15)
                face_rgb = rgb[y1:y2, x1:x2]

                if face_rgb.size == 0:
                    continue

                emb = self._embed(face_rgb)
                if emb is None:
                    continue

                name, sim = self._identify(emb)
                detected_names.append(name)

                just_marked = self._update_timer(name)
                if just_marked:
                    newly_marked.append(name)

                # Colour: green=marked, yellow=accumulating, red=unknown
                if name in self.marked:
                    color = (0, 200, 0)
                elif name != "Unknown":
                    elapsed = time.time() - self.first_seen.get(name, time.time())
                    pct = min(elapsed / TIME_THRESHOLD_SEC, 1.0)
                    color = (0, int(255 * pct), 255 - int(255 * pct))
                else:
                    color = (0, 0, 220)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({sim:.2f})"
                if name in self.marked:
                    label += " ✓"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Reset timers for students not seen this frame
        seen_set = set(detected_names)
        for name in list(self.first_seen.keys()):
            if name not in seen_set:
                self.reset_timer(name)

        return annotated, newly_marked

    def get_status(self):
        """Return dict with marked students and timer progress."""
        now = time.time()
        timers = {
            name: min((now - ts) / TIME_THRESHOLD_SEC * 100, 100)
            for name, ts in self.first_seen.items()
            if name != "Unknown"
        }
        return {"marked": list(self.marked), "timers": timers}