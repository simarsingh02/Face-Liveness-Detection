import argparse
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# Import your model logic
from src.face_detection import detect_face
from src.label_dataset import label_images
from src.vit_model import Patches, PatchEncoder # Necessary for ViT loading
from src.lbp_feature import extract_lbp
import joblib

# Use the EXACT same preprocess function from main.py
def preprocess_face(face, target_size=(128, 128)):
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    if len(face.shape) == 2:
        face = np.expand_dims(face, axis=-1)
    return face

def load_dataset(dataset_path):
    print(f"[*] Loading test dataset: {dataset_path}")
    image_paths, labels = label_images(dataset_path)
    raw_faces, y = [], []
    for img_path, label in zip(image_paths, labels):
        image = cv2.imread(img_path)
        if image is None: continue
        face = detect_face(image)
        if face is None: continue
        raw_faces.append(face)
        y.append(label)
    return raw_faces, np.array(y, dtype="float32")

def compute_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(np.int32)
    far = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
    frr = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": accuracy, "far": far, "frr": frr, "hter": (far + frr) / 2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/CASIA/Test/Colour")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Define models to check
    MODELS_TO_EVALUATE = [
        {"name": "CNN", "path": "models/liveness_cnn.keras", "type": "keras", "size": (128, 128)},
        {"name": "Hybrid", "path": "models/liveness_hybrid.keras", "type": "keras", "size": (128, 128)},
        {"name": "ViT", "path": "models/liveness_vit.keras", "type": "keras", "size": (128, 128)}, 
        {"name": "LBP+SVM", "path": "models/liveness_lbp_svm.pkl", "type": "svm", "size": (128, 128)},
    ]

    raw_faces, y_test = load_dataset(args.dataset)
    results = []

    for m_info in MODELS_TO_EVALUATE:
        path = Path(m_info["path"])
        if not path.exists():
            print(f"[!] {m_info['name']} not found at {path}")
            continue

        print(f"[*] Evaluating {m_info['name']}...")
        
        try:
            if m_info.get("type") == "svm":
                # 1. Load Model
                model = joblib.load(str(path))
                
                # 2. Extract Features
                X_features = []
                for f in raw_faces:
                    if len(f.shape) == 3 and f.shape[2] == 3:
                        gray_face = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_face = f
                    X_features.append(extract_lbp(gray_face))
                X_test = np.array(X_features)
                
                # 3. Predict and Score
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                # 1. Preprocess using model-specific size
                X_test = np.array([preprocess_face(f, target_size=m_info["size"]) for f in raw_faces])

                # 2. Load with Custom Objects (Crucial for ViT)
                model = tf.keras.models.load_model(str(path), custom_objects={
                    "Patches": Patches, 
                    "PatchEncoder": PatchEncoder
                })
                
                # 3. Predict and Score
                y_scores = model.predict(X_test, verbose=0).ravel()

            metrics = compute_metrics(y_test, y_scores, threshold=args.threshold)
            results.append({"name": m_info["name"], "metrics": metrics})
        except Exception as e:
            print(f"[X] Error: {e}")

    # Print Table (Simplified)
    print("\n| Model   | Accuracy | FAR | FRR | HTER |")
    print("|---------|----------|-----|-----|------|")
    for r in results:
        m = r["metrics"]
        print(f"| {r['name']:<7} | {m['accuracy']:.2%} | {m['far']:.2%} | {m['frr']:.2%} | {m['hter']:.2%} |")

if __name__ == "__main__":
    main()