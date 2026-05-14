import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.face_detection import detect_face
from src.hybrid_model import load_hybrid_model

def preprocess_face(face):
    face = cv2.resize(face, (128, 128))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    return face

def main():
    parser = argparse.ArgumentParser(description="Face Liveness Inference on a Single Image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/liveness_hybrid.keras",
        help="Path to the trained .keras model (default: models/liveness_hybrid.keras)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")

    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found at '{model_path}'")
        sys.exit(1)

    # 1. Read Image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        sys.exit(1)

    # 2. Detect Face
    face = detect_face(image)
    if face is None:
        print("Error: No face detected in the image.")
        sys.exit(1)

    # 3. Preprocess Face
    processed_face = preprocess_face(face)

    # 4. Load Model
    print(f"Loading model from '{model_path}'...")
    # load_hybrid_model passes necessary custom objects (like Patches) for ViT/Hybrid
    # and safely ignores them if loading a pure CNN model.
    model = load_hybrid_model(str(model_path))

    # 5. Predict
    print("Running inference...")
    score = model.predict(processed_face, verbose=0)[0][0]

    # 6. Output Result
    prediction = "Live" if score >= args.threshold else "Spoof"
    
    print("-" * 30)
    print("Inference Result:")
    print(f"Image       : {image_path.name}")
    print(f"Model       : {model_path.name}")
    print(f"Score       : {score:.4f}")
    print(f"Prediction  : {prediction}")
    print("-" * 30)

    # 7. Display Result Image
    display_image = image.copy()
    text = f"{prediction}: {score:.4f}"
    # Color in BGR: Green for Live, Red for Spoof
    color = (0, 255, 0) if prediction == "Live" else (0, 0, 255)
    
    cv2.putText(
        display_image, 
        text, 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        color, 
        2, 
        cv2.LINE_AA
    )

    try:
        import matplotlib.pyplot as plt
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(display_image_rgb)
        plt.title(f"Inference Result: {prediction}")
        plt.axis("off")
        plt.show()
    except ImportError:
        cv2.imshow("Inference Result", display_image)
        print("Press any key on the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
