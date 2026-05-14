import numpy as np
from skimage.feature import local_binary_pattern
import cv2

def extract_lbp(face):

    face = cv2.resize(face, (128,128))

    radius = 1
    points = 8 * radius

    block_size = 16

    features = []

    for y in range(0, 128, block_size):
        for x in range(0, 128, block_size):

            block = face[y:y+block_size, x:x+block_size]

            lbp = local_binary_pattern(block, points, radius, method="uniform")

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, points+3),
                range=(0, points+2)
            )

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)

            features.extend(hist)

    return np.array(features)