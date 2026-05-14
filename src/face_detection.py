import cv2

cascade_path = "haarcascade/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return gray, faces


def detect_face(image):
    gray, faces = detect_faces(image)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        return face

    return None
