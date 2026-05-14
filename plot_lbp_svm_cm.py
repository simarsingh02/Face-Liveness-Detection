import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from compare_models import load_dataset
from src.lbp_feature import extract_lbp

def main():
    test_dataset_path = "c:/Projects/Face_Liveness_Project/dataset/CASIA/Test/Colour"
    print("Loading dataset...")
    raw_faces, y_test = load_dataset(test_dataset_path)
    
    print("Loading model...")
    model = joblib.load("c:/Projects/Face_Liveness_Project/models/liveness_lbp_svm.pkl")
    
    print("Extracting features...")
    X_features = []
    for f in raw_faces:
        if len(f.shape) == 3 and f.shape[2] == 3:
            gray_face = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = f
        X_features.append(extract_lbp(gray_face))
    X_test = np.array(X_features)
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    print("Plotting...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Live'])
    disp.plot(cmap='Blues', colorbar=False)
    plt.title('LBP+SVM Confusion Matrix (Test Data)')
    plt.tight_layout()
    
    save_path = "c:/Projects/Face_Liveness_Project/lbp_svm_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")

if __name__ == "__main__":
    main()
