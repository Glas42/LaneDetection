import bettercam
import keyboard
import numpy
import time
import cv2
import mss
import os

capture_time = 5
pause_key = "f"

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\RawDataset"

while True:
    user_input = input(f"Information: This script will capture your screen every {capture_time} seconds, do you want to continue? (y/n)\n-> ").lower()
    if user_input == 'y':
        break
    elif user_input == 'n':
        exit()

sct = mss.mss()
screen_width = sct.monitors[(1)]["width"]
screen_height = sct.monitors[(1)]["height"]
empty_frame = numpy.zeros((screen_height, screen_width, 3), numpy.uint8)
camera = bettercam.create(output_color="BGR", output_idx=0)
last_frame = empty_frame.copy()
last_captured = time.time()
last_pressed = False
capturing = False

cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Data Collection", cv2.WND_PROP_TOPMOST, 1)
cv2.imshow("Data Collection", empty_frame)

while True:
    start = time.time()

    if capturing and time.time() - last_captured > capture_time:
        last_captured = time.time()
        frame = camera.grab()
        if frame is None:
            continue
        cv2.imwrite(f"{DATA_PATH}\\{len([name for name in os.listdir(DATA_PATH) if name.endswith('.png')])}.png", frame)
        last_frame = frame.copy()

    try:
        _, _, _, _ = cv2.getWindowImageRect("Data Collection")
    except:
        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Data Collection", cv2.WND_PROP_TOPMOST, 1)

    if capturing:
        cv2.imshow("Data Collection", last_frame)
    else:
        cv2.imshow("Data Collection", empty_frame)

    cv2.waitKey(1)

    pressed = keyboard.is_pressed(pause_key)
    if pressed == True and last_pressed == False:
        capturing = not capturing
    last_pressed = pressed

    time_to_sleep = 1/60 - (time.time() - start)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)