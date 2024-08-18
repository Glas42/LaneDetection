import numpy as np
from ctypes import windll, byref, sizeof, c_int
import win32gui, win32con
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\PreprocessedDataset"

print("Caching images...")

images = []
for file in os.listdir(DATA_PATH):
    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and os.path.exists(f"{PATH}\\Datasets\\FinalDataset\\{file.replace(file.split('.')[-1], 'txt')}") == False:
        image = cv2.imread(f"{DATA_PATH}\\{file}", cv2.IMREAD_UNCHANGED)
        images.append(image)

print("Done!")

def CreateWindow():
    cv2.namedWindow("LaneDetection - Annotation", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LaneDetection - Annotation", cv2.WND_PROP_TOPMOST, 1)
    if os.name == 'nt':
        hwnd = win32gui.FindWindow(None, "LaneDetection - Annotation")
        windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, byref(c_int(0x000000)), sizeof(c_int))
        icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        hicon = win32gui.LoadImage(None, f"{PATH}\\icon.ico", win32con.IMAGE_ICON, 0, 0, icon_flags)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, hicon)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, hicon)

index = 0

while index < len(images):
    try:
        _, _, _, _ = cv2.getWindowImageRect("LaneDetection - Annotation")
    except:
        CreateWindow()

    cv2.imshow("LaneDetection - Annotation", images[index])
    cv2.waitKey(0)
    index += 1