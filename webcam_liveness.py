from pathlib import Path

import cv2
import numpy as np

from src.face_detection import detect_faces
from src.hybrid_model import load_hybrid_model


MODEL_PATH = Path("models/liveness_hybrid.keras")


def preprocess_face(face):
    face = cv2.resize(face, (128, 128))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}", flush=True)
        print("Train the hybrid model first by running: python main.py", flush=True)
        return

    print(f"Loading model from: {MODEL_PATH}", flush=True)
    model = load_hybrid_model(MODEL_PATH)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.", flush=True)
        return

    print("Webcam started. Press 'q' to quit.", flush=True)

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read frame from webcam.", flush=True)
            break

        gray, faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_input = preprocess_face(face)

            confidence = float(model.predict(face_input, verbose=0)[0][0])
            label = "Live" if confidence >= 0.5 else "Spoof"
            color = (0, 255, 0) if label == "Live" else (0, 0, 255)

            text = f"{label} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
