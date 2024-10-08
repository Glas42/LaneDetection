import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_SRC_PATH = PATH + "\\Datasets\\RawDataset"
DATA_DST_PATH = PATH + "\\Datasets\\PreprocessedDataset"

tilt_image = False
normalize = False
resolution = 400

def vertical_perspective_warp(image, angle):
    height, width = image.shape[:2]
    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_pts = np.float32([[0, 0], [width - 1, 0], [int(width * np.sin(np.radians(angle))), height - 1], [width - 1 - int(width * np.sin(np.radians(angle))), height - 1]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    return warped_image

print("Preprocessing...")

for image in os.listdir(DATA_SRC_PATH):
    image = cv2.imread(os.path.join(DATA_SRC_PATH, image))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if tilt_image:
        image = vertical_perspective_warp(image, 26.5)

    if normalize:
        avg_color = 70
        mean_color = np.mean(image)
        if mean_color != avg_color:
            scaling_factor = avg_color / mean_color
            image = cv2.multiply(image, scaling_factor)
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 20), -4, 0)

    image = cv2.resize(image, (resolution, resolution))

    cv2.imwrite(f"{DATA_DST_PATH}\\{len([name for name in os.listdir(DATA_DST_PATH) if name.endswith('.png')])}.png", image)

print("Done!")