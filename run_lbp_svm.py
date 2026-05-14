import argparse
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.face_detection import detect_face
from src.label_dataset import label_images
from src.lbp_feature import extract_lbp

def load_dataset(dataset_path):
    print(f"[*] Loading dataset: {dataset_path}")
    image_paths, labels = label_images(dataset_path)
    raw_faces, y = [], []
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(img_path)
        if image is None: continue
        face = detect_face(image)
        if face is None: continue
        raw_faces.append(face)
        y.append(label)
        
        if (i+1) % 100 == 0 or (i+1) == len(image_paths):
            print(f"    Processed {i+1}/{len(image_paths)} images...", end='\r', flush=True)
            
    print(f"\n[+] Total valid faces detected: {len(raw_faces)}")
    return raw_faces, np.array(y, dtype="float32")

def preprocess_and_extract_lbp(faces):
    print(f"[*] Extracting LBP features from {len(faces)} faces...")
    X_features = []
    for face in faces:
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # extract_lbp automatically resizes to 128x128
        features = extract_lbp(face)
        X_features.append(features)
        
    return np.array(X_features)

def compute_metrics(y_true, y_pred):
    far = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
    frr = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": accuracy, "far": far, "frr": frr, "hter": (far + frr) / 2}

def main():
    train_dataset_path = "dataset/CASIA/Train/Colour"
    test_dataset_path = "dataset/CASIA/Test/Colour"
    model_path = Path("models/liveness_lbp_svm.pkl")

    # 1. Load Data
    raw_train_faces, y_train = load_dataset(train_dataset_path)
    raw_test_faces, y_test = load_dataset(test_dataset_path)

    # 2. Extract Features
    X_train = preprocess_and_extract_lbp(raw_train_faces)
    X_test = preprocess_and_extract_lbp(raw_test_faces)

    print(f"\n[*] Training SVM Classifier on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    # 3. Train SVM
    # using class_weight='balanced' to handle any class imbalance
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Save Model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[+] Saved SVM model to {model_path}")

    # 5. Evaluate
    print("[*] Evaluating SVM on Test Set...")
    y_pred = clf.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    
    print("\n--- LBP + SVM Results ---")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"FAR:      {metrics['far']:.2%}")
    print(f"FRR:      {metrics['frr']:.2%}")
    print(f"HTER:     {metrics['hter']:.2%}")
    
if __name__ == "__main__":
    main()
