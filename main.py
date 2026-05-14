from src.label_dataset import label_images
from src.face_detection import detect_face
from src.classifier import train_model

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_face(face, target_size=(128, 128)):
    # 1. Convert to grayscale. 
    # Since the CASIA folder is 'Colour', images have 3 channels.
    # Your Hybrid model expects 1 channel (128, 128, 1).
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # 2. Resize to the model's expected dimensions
    face = cv2.resize(face, target_size)
    
    # 3. Normalize pixel values to [0, 1]
    face = face.astype("float32") / 255.0
    
    # 4. CRITICAL: Ensure the shape is (H, W, 1) and not just (H, W)
    if len(face.shape) == 2:
        face = np.expand_dims(face, axis=-1)
        
    return face


def load_dataset(dataset_path):
    print(f"[*] Loading dataset: {dataset_path}", flush=True)
    image_paths, labels = label_images(dataset_path)
    
    raw_faces = [] 
    y = []

    for index, (img_path, label) in enumerate(zip(image_paths, labels), start=1):
        image = cv2.imread(img_path)
        if image is None: continue
        
        face = detect_face(image)
        if face is None: continue

        # Just append the raw crop, no preprocessing yet!
        raw_faces.append(face)
        y.append(label)

        # Print progress to know it hasn't frozen
        if index % 100 == 0 or index == len(image_paths):
            print(f"    Processed {index}/{len(image_paths)} images...", end='\r', flush=True)
            
    print(f"\n[+] Total valid faces detected: {len(raw_faces)}")
    return raw_faces, np.array(y, dtype="float32")


def main():
    # 1. Define paths
    train_dataset_path = "dataset/CASIA/Train/Colour"
    test_dataset_path = "dataset/CASIA/Test/Colour"
    
    # SETTINGS: (128, 128) for Hybrid/CNN, (224, 224) for ViT
    target_size = (128, 128) 

    # 2. Load the raw faces (returns list of faces and numpy array of labels)
    raw_train_faces, y_train_full = load_dataset(train_dataset_path)
    raw_test_faces, y_test = load_dataset(test_dataset_path)

    # 3. Preprocess the faces to the target size
    print(f"\n[*] Preprocessing images to {target_size}...", flush=True)

    X_train_full = np.array([preprocess_face(f, target_size=target_size) for f in raw_train_faces])
    X_test = np.array([preprocess_face(f, target_size=target_size) for f in raw_test_faces])
    
    # Ensure dimensions are exactly 4D: (Samples, Height, Width, Channels)
    # If preprocess_face somehow returns an extra batch dim, squeeze it out
    if len(X_train_full.shape) == 5:
        X_train_full = np.squeeze(X_train_full, axis=1)
        X_test = np.squeeze(X_test, axis=1)
        
    # If the array stacked into 3D (Samples, Height, Width), add the channel back
    if len(X_train_full.shape) == 3:
        X_train_full = np.expand_dims(X_train_full, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # 4. Split into Train and Validation
    print("[*] Splitting data into Training and Validation sets...", flush=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"\n--- Data Summary ---")
    print(f"Train data shape:      {X_train.shape}")
    print(f"Train labels shape:    {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape:       {X_test.shape}")
    print(f"Test labels shape:     {y_test.shape}")
    print(f"--------------------\n")

    # 5. Start training
    model_type = "cnn"  # Change this to "hybrid" or "cnn" as needed
    print(f"[*] Initiating model training for {model_type}...")
    # model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_type="hybrid")
    model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_type=model_type)

    # 6. Plot Training Curves and Confusion Matrix
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
        
        print("[*] Plotting training curves and confusion matrix...")
        plt.figure(figsize=(18, 5))
        
        # Accuracy Plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Confusion Matrix
        ax = plt.subplot(1, 3, 3)
        print("[*] Generating predictions for confusion matrix...")
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = (y_pred >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Live'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title('Confusion Matrix (Test Data)')

        # Classification Report — Precision, Recall, F1 for each class
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred_classes, target_names=['Spoof', 'Live']))
        
        plt.tight_layout()
        save_path = f'{model_type}_training_curves.png'
        plt.savefig(save_path)
        print(f"[+] Saved training curves and confusion matrix to '{save_path}'")
        plt.show()
    except ImportError as e:
        print(f"[!] Required library not installed: {e}. Skipping plots.")

if __name__ == "__main__":
    main()