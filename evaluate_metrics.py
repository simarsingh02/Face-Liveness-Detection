from pathlib import Path

import cv2
import numpy as np

from src.face_detection import detect_face
from src.hybrid_model import load_hybrid_model
from src.label_dataset import label_images


MODEL_PATH = Path("models/liveness_hybrid.keras")
TEST_DATASET_PATH = "dataset/CASIA/Test/Colour"
THRESHOLD = 0.5


def preprocess_face(face):
    face = cv2.resize(face, (128, 128))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    return face


def load_dataset(dataset_path):
    print(f"Loading evaluation dataset from: {dataset_path}", flush=True)

    image_paths, labels = label_images(dataset_path)
    print(f"Images found: {len(image_paths)}", flush=True)

    X = []
    y = []

    for index, (img_path, label) in enumerate(zip(image_paths, labels), start=1):
        image = cv2.imread(img_path)

        if image is None:
            continue

        face = detect_face(image)

        if face is None:
            continue

        X.append(preprocess_face(face))
        y.append(label)

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)} images...", flush=True)

    X = np.array(X)
    y = np.array(y, dtype="float32")

    print(f"Total valid evaluation samples: {len(X)}", flush=True)
    return X, y


def compute_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    false_accepts = np.sum((y_true == 0) & (y_pred == 1))
    false_rejects = np.sum((y_true == 1) & (y_pred == 0))
    total_attacks = np.sum(y_true == 0)
    total_real = np.sum(y_true == 1)

    far = false_accepts / total_attacks if total_attacks else 0.0
    frr = false_rejects / total_real if total_real else 0.0
    hter = (far + frr) / 2.0

    return far, frr, hter, false_accepts, false_rejects, total_attacks, total_real


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}", flush=True)
        print("Train the hybrid model first by running: python main.py", flush=True)
        return

    X_test, y_test = load_dataset(TEST_DATASET_PATH)

    print(f"Loading model from: {MODEL_PATH}", flush=True)
    model = load_hybrid_model(MODEL_PATH)

    y_scores = model.predict(X_test, verbose=0).ravel()

    far, frr, hter, false_accepts, false_rejects, total_attacks, total_real = compute_metrics(
        y_test,
        y_scores,
        threshold=THRESHOLD,
    )

    print("", flush=True)
    print(f"Threshold: {THRESHOLD:.2f}", flush=True)
    print(f"Attack samples (spoof): {total_attacks}", flush=True)
    print(f"Real samples (live): {total_real}", flush=True)
    print(f"False Accepts: {false_accepts}", flush=True)
    print(f"False Rejects: {false_rejects}", flush=True)
    print(f"FAR: {far:.4f} ({far * 100:.2f}%)", flush=True)
    print(f"FRR: {frr:.4f} ({frr * 100:.2f}%)", flush=True)
    print(f"HTER: {hter:.4f} ({hter * 100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
