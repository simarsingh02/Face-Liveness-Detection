import os
import cv2

def load_images(dataset_path):
    images = []

    for img_name in os.listdir(dataset_path):

        img_path = os.path.join(dataset_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        images.append(image)

    print("Total images loaded:", len(images))

    return images