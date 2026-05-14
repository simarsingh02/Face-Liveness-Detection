import os

def label_images(dataset_path):

    image_paths = []
    labels = []

    for img_name in os.listdir(dataset_path):

        # only process images
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(dataset_path, img_name)

        # check filename for label
        if "real" in img_name.lower():
            label = 1   # live face
        else:
            label = 0   # spoof face

        image_paths.append(img_path)
        labels.append(label)

    return image_paths, labels