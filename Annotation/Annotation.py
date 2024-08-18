import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\PreprocessedDataset"

print("Caching images...")

images = []
for file in os.listdir(DATA_PATH):
    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and os.path.exists(f"{DATA_PATH}\\{file.replace(file.split('.')[-1], 'txt')}") == False:
        image = cv2.imread(f"{DATA_PATH}\\{file}", cv2.IMREAD_UNCHANGED)
        images.append(image)

print("Done!")

