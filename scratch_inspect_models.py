import os
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

from src.face_detection import detect_face
from src.label_dataset import label_images
from src.hybrid_model import Patches, PatchEncoder

def main():
    print("Loading test dataset...")
    image_paths, labels = label_images("dataset/CASIA/Test/Colour")
    
    # Take a small subset just to see how long it takes
    print(f"Total test images: {len(image_paths)}")
    
    try:
        with open("models/liveness_model.pkl", "rb") as f:
            clf = pickle.load(f)
            print("Successfully loaded liveness_model.pkl:", type(clf))
    except Exception as e:
        print("Failed to load liveness_model.pkl:", e)

if __name__ == "__main__":
    main()
