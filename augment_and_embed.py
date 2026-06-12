"""
Step 2: Data augmentation on raw face images, then generate FaceNet embeddings.
Run once after collecting all student images.
Usage: python augment_and_embed.py
"""

import os
import pickle
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR      = "train_images"
EMBED_DIR      = "embeddings"
EMBED_FILE     = os.path.join(EMBED_DIR, "embeddings.pkl")
IMG_SIZE       = (160, 160)   # FaceNet input size
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EMBED_DIR, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading FaceNet on {DEVICE} ...")
mtcnn   = MTCNN(image_size=160, margin=20, device=DEVICE, keep_all=False)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ── Augmentation helpers ──────────────────────────────────────────────────────
def augment(img: Image.Image):
    """Return list of augmented PIL images from one source image."""
    variants = [img]  # original

    # Horizontal flip
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Brightness variations
    for factor in [0.6, 1.4]:
        variants.append(ImageEnhance.Brightness(img).enhance(factor))

    # Contrast variation
    variants.append(ImageEnhance.Contrast(img).enhance(1.3))

    # Slight blur (simulates distance)
    variants.append(img.filter(ImageFilter.GaussianBlur(radius=1)))

    # Rotation variations
    for angle in [-10, 10]:
        variants.append(img.rotate(angle, expand=False, fillcolor=(0, 0, 0)))

    return variants


def get_embedding(img: Image.Image):
    """Return 512-d FaceNet embedding for a PIL image, or None if no face found."""
    tensor = mtcnn(img)
    if tensor is None:
        return None
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = facenet(tensor).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)   # L2-normalise for cosine similarity


# ── Main loop ─────────────────────────────────────────────────────────────────
all_embeddings = []
all_labels     = []

students = [d for d in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, d))]

if not students:
    print("[ERROR] No student folders found in train_images/. Run collect_faces.py first.")
    exit(1)

for student in students:
    student_dir = os.path.join(TRAIN_DIR, student)
    images = [f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\n[{student}] Processing {len(images)} images ...")

    ok = 0
    for fname in images:
        path = os.path.join(student_dir, fname)
        try:
            img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        for aug_img in augment(img):
            emb = get_embedding(aug_img)
            if emb is not None:
                all_embeddings.append(emb)
                all_labels.append(student)
                ok += 1

    print(f"  → {ok} embeddings generated (with augmentation)")

# ── Save ──────────────────────────────────────────────────────────────────────
data = {"embeddings": np.array(all_embeddings), "labels": all_labels}
with open(EMBED_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"\n[DONE] Saved {len(all_labels)} embeddings → {EMBED_FILE}")
print(f"       Students: {list(set(all_labels))}")