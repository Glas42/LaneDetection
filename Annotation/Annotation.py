import numpy as np
import ctypes
import mouse
import cv2
import os

if os.name == 'nt':
    from ctypes import windll, byref, sizeof, c_int
    import win32gui, win32con

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\PreprocessedDataset"

index = 0
max_lanes = 3
image_scale = 2
last_left_clicked = False
last_window_size = None, None

lane_buttons = [0] * (max_lanes * 2 + 1)

print("Caching images...")

images = []
for file in os.listdir(DATA_PATH):
    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and os.path.exists(f"{PATH}\\Datasets\\FinalDataset\\{file.replace(file.split('.')[-1], 'txt')}") == False:
        image = cv2.imread(f"{DATA_PATH}\\{file}", cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.resize(image, (round(image.shape[1] * image_scale), round(image.shape[0] * image_scale)))
        images.append(image)

print("Done!")

def CreateWindow():
    cv2.namedWindow("LaneDetection - Annotation", cv2.WINDOW_NORMAL)
    if os.name == 'nt':
        hwnd = win32gui.FindWindow(None, "LaneDetection - Annotation")
        windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, byref(c_int(0x000000)), sizeof(c_int))
        icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        hicon = win32gui.LoadImage(None, f"{PATH}\\icon.ico", win32con.IMAGE_ICON, 0, 0, icon_flags)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, hicon)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, hicon)
CreateWindow()

def GetTextSize(text="NONE", text_width=100, max_text_height=100):
    fontscale = 1
    textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
    width_current_text, height_current_text = textsize
    max_count_current_text = 3
    while width_current_text != text_width or height_current_text > max_text_height:
        fontscale *= min(text_width / textsize[0], max_text_height / textsize[1])
        textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
        max_count_current_text -= 1
        if max_count_current_text <= 0:
            break
    thickness = round(fontscale * 2)
    if thickness <= 0:
        thickness = 1
    return text, fontscale, thickness, textsize[0], textsize[1]


def Button(text="NONE", x1=0, y1=0, x2=100, y2=100, round_corners=30, buttoncolor=(100, 100, 100), buttonhovercolor=(130, 130, 130), buttonselectedcolor=(160, 160, 160), buttonselectedhovercolor=(190, 190, 190), buttonselected=False, textcolor=(255, 255, 255), width_scale=0.9, height_scale=0.8):
    if x1 <= mouse_x*frame_width <= x2 and y1 <= mouse_y*frame_height <= y2:
        buttonhovered = True
    else:
        buttonhovered = False
    if buttonselected == True:
        if buttonhovered == True:
            cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedhovercolor, round_corners, cv2.LINE_AA)
            cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedhovercolor, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedcolor, round_corners, cv2.LINE_AA)
            cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonselectedcolor, -1, cv2.LINE_AA)
    elif buttonhovered == True:
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonhovercolor, round_corners, cv2.LINE_AA)
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttonhovercolor, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttoncolor, round_corners, cv2.LINE_AA)
        cv2.rectangle(frame, (round(x1+round_corners/2), round(y1+round_corners/2)), (round(x2-round_corners/2), round(y2-round_corners/2)), buttoncolor, -1, cv2.LINE_AA)
    text, fontscale, thickness, width, height = GetTextSize(text, round((x2-x1)*width_scale), round((y2-y1)*height_scale))
    cv2.putText(frame, text, (round(x1 + (x2-x1) / 2 - width / 2), round(y1 + (y2-y1) / 2 + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor, thickness, cv2.LINE_AA)
    if x1 <= mouse_x*frame_width <= x2 and y1 <= mouse_y*frame_height <= y2 and left_clicked == False and last_left_clicked == True:
        return True, buttonhovered
    else:
        return False, buttonhovered

while index < len(images):
    try:
        window_x, window_y, window_width, window_height = cv2.getWindowImageRect("LaneDetection - Annotation")
        if window_width != last_window_size[0] or window_height != last_window_size[1]:
            last_window_size = window_width, window_height
            background = np.zeros((window_height, window_width, 3), np.uint8)
        mouse_x, mouse_y = mouse.get_position()
        mouse_relative_window = mouse_x - window_x, mouse_y - window_y
        if window_width != 0 and window_height != 0:
            mouse_x = mouse_relative_window[0]/window_width
            mouse_y = mouse_relative_window[1]/window_height
        else:
            mouse_x = 0
            mouse_y = 0
    except:
        exit()

    if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "LaneDetection - Annotation"):
        left_clicked = True
    else:
        left_clicked = False

    frame = background.copy()
    frame_height, frame_width, _ = frame.shape
    frame[0:background.shape[0], 0:round(background.shape[1] * 0.7)] = cv2.resize(cv2.cvtColor(images[index], cv2.COLOR_GRAY2BGR), (round(background.shape[1] * 0.7), background.shape[0]))

    button_next_pressed, button_next_hovered = Button(text="Next",
                                                        x1=0.705*frame_width,
                                                        y1=0.01*frame_height,
                                                        x2=0.995*frame_width,
                                                        y2=0.11*frame_height,
                                                        round_corners=30,
                                                        buttoncolor=(0, 200, 0),
                                                        buttonhovercolor=(20, 220, 20),
                                                        buttonselectedcolor=(20, 220, 20),
                                                        textcolor=(255, 255, 255),
                                                        width_scale=0.95,
                                                        height_scale=0.5)

    for button in range(len(lane_buttons)):
        button_pressed, button_hovered = Button(text=f"Lane {int(button - (len(lane_buttons) - 1) / 2)}",
                                                            x1=0.705*frame_width,
                                                            y1=((button + 1) / (len(lane_buttons) + 1) + 0.01) * frame_height,
                                                            x2=0.995*frame_width,
                                                            y2=((button + 1) / (len(lane_buttons) + 1) + 0.11) * frame_height,
                                                            round_corners=30,
                                                            buttonselected=lane_buttons[button] == 1,
                                                            buttoncolor=(80, 80, 80),
                                                            buttonhovercolor=(100, 100, 100),
                                                            buttonselectedcolor=(0, 200, 0),
                                                            buttonselectedhovercolor=(20, 220, 20),
                                                            textcolor=(255, 255, 255),
                                                            width_scale=0.95,
                                                            height_scale=0.5)
        if button_pressed == True:
            lane_buttons[button] = 1 if lane_buttons[button] == 0 else 0

    if button_next_pressed == True:
        index += 1

    last_left_clicked = left_clicked

    cv2.imshow("LaneDetection - Annotation", frame)
    cv2.waitKey(1)