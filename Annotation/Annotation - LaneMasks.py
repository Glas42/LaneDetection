import numpy as np
import shutil
import ctypes
import mouse
import math
import time
import cv2
import os

if os.name == 'nt':
    from ctypes import windll, byref, sizeof, c_int
    import win32gui, win32con

PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\PreprocessedDataset"
DST_PATH = PATH + "\\Datasets\\FinalDataset"

fps = 60
max_lanes = 0 # amount of lanes from center lane to left and right

index = 0
remove_list = []
last_left_clicked = False
last_right_clicked = False
allow_adding_points = True
last_window_size = None, None
grabbed_point = None, None, None

lane_buttons = [0] * (max_lanes * 2 + 1)
lanes = [[[(0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) - 1 / ((max_lanes * 2 + 2) * 2),   0,    False), # x, y, tied_to_edge
           (0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) - 1 / ((max_lanes * 2 + 2) * 2),   0.5,  False),
           (0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) - 1 / ((max_lanes * 2 + 2) * 2),   1,    True)
           ],
          [(0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) + 1 / ((max_lanes * 2 + 2) * 2),   0,    False),
           (0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) + 1 / ((max_lanes * 2 + 2) * 2),   0.5,  False),
           (0.5 - (i - (len(lane_buttons) - 1) / 2) / (max_lanes * 2 + 1) + 1 / ((max_lanes * 2 + 2) * 2),   1,    True)
           ]] for i in range(max_lanes * 2 + 1)]
lanes = [[[((np.clip(x, 0, 1), y, tied_to_edge)) for x, y, tied_to_edge in lane] for lane in lanes_group] for lanes_group in lanes]

print("Caching images...")

images = []
i = 0
for file in os.listdir(DATA_PATH):
    exists = False
    for i in range(15):
        if os.path.exists(f"{DST_PATH}\\{file.split('.')[0].split('#')[0]}#{i - 7}.{file.split('.')[-1]}") == True:
            exists = True
    if exists == False:
        image = cv2.imread(f"{DATA_PATH}\\{file}", cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.resize(image, (image.shape[1], image.shape[0]))
        images.append((image, file))

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
    start = time.time()

    try:
        window_x, window_y, window_width, window_height = cv2.getWindowImageRect("LaneDetection - Annotation")
        if window_width < 50 or window_height < 50:
            cv2.resizeWindow("LaneDetection - Annotation", 50, 50)
            window_width = 50
            window_height = 50
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
        mouse_x_image = mouse_x / 0.7
        mouse_y_image = mouse_y
    except:
        exit()

    if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "LaneDetection - Annotation"):
        left_clicked = True
    else:
        left_clicked = False

    if ctypes.windll.user32.GetKeyState(0x02) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "LaneDetection - Annotation"):
        right_clicked = True
    else:
        right_clicked = False

    frame = background.copy()
    frame_height, frame_width, _ = frame.shape
    image = cv2.resize(cv2.cvtColor(images[index][0], cv2.COLOR_GRAY2BGR), (round(background.shape[1] * 0.7), background.shape[0]))
    cv2.line(image, (round(0.5 * image.shape[1]), 0), (round(0.5 * image.shape[1]), image.shape[0]), (0, 0, 100), 1, cv2.LINE_AA)

    button_next_pressed, button_next_hovered = Button(text="Next",
                                                      x1=0.85125*frame_width,
                                                      y1=0.005 * frame_height,
                                                      x2=0.9975*frame_width,
                                                      y2=(1 / (len(lane_buttons) + 1) - 0.005) * frame_height,
                                                      round_corners=30,
                                                      buttoncolor=(0, 200, 0),
                                                      buttonhovercolor=(20, 220, 20),
                                                      buttonselectedcolor=(20, 220, 20),
                                                      textcolor=(255, 255, 255),
                                                      width_scale=0.95,
                                                      height_scale=0.5)

    button_back_pressed, button_back_hovered = Button(text="Back",
                                                      x1=0.7025*frame_width,
                                                      y1=0.005*frame_height,
                                                      x2=0.84825*frame_width,
                                                      y2=(1 / (len(lane_buttons) + 1) - 0.005) * frame_height,
                                                      round_corners=30,
                                                      buttoncolor=(0, 0, 200),
                                                      buttonhovercolor=(20, 20, 220),
                                                      buttonselectedcolor=(20, 20, 220),
                                                      textcolor=(255, 255, 255),
                                                      width_scale=0.95,
                                                      height_scale=0.5)

    if button_next_pressed == True and index < len(images) - 1:
        try:
            shutil.copy2(f"{DATA_PATH}\\{images[index][1]}", f"{DST_PATH}\\{images[index][1]}")

            for lane in range(len(lane_buttons)):
                export_image = np.zeros((cv2.imread(f"{DATA_PATH}\\{images[index][1]}").shape[0], cv2.imread(f"{DATA_PATH}\\{images[index][1]}").shape[1]), np.uint8)

                if lane_buttons[lane] == 1:
                    left_points = []
                    right_points = []
                    for i in range(2):
                        # check if there is a point with y = 1 in the points, if not add a point, the x of that point will be the x of the lowest original point
                        has_y1_point = False
                        lowest_x = None
                        lowest_y = None
                        for j in range(len(lanes[lane][i])):
                            x, y, tied_to_edge = lanes[lane][i][j]
                            if y == 1:
                                has_y1_point = True
                            if lowest_x is None or y > lowest_y:
                                lowest_x = x
                                lowest_y = y
                        if not has_y1_point:
                            lanes[lane][i].append((lowest_x, 1, "temp_point"))
                        for j in range(len(lanes[lane][i])):
                            x, y, tied_to_edge = lanes[lane][i][j]
                            if i == 0:
                                left_points.append((round(x * export_image.shape[1]), round(y * export_image.shape[0])))
                            else:
                                right_points.append((round(x * export_image.shape[1]), round(y * export_image.shape[0])))
                    right_points.reverse()
                    points = []
                    for point in left_points:
                        points.append(point)
                    for point in right_points:
                        points.append(point)
                    cv2.fillPoly(export_image, np.array([points], dtype=np.int32), (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite(f"{DST_PATH}\\{str(images[index][1]).replace(str(images[index][1]).split('.')[-1], '').replace('.', '')}#{lane - max_lanes}.png", export_image)
        except:
            import traceback
            traceback.print_exc()
        for lane in range(len(lane_buttons)):
            if lane_buttons[lane] == 1:
                for i in range(2):
                    while len(lanes[lane][i]) > 0 and lanes[lane][i][-1][2] == "temp_point":
                        lanes[lane][i].pop()
        index += 1

    if button_back_pressed == True and index > 0:
        index -= 1

    for button in range(len(lane_buttons)):
        button_pressed, button_hovered = Button(text=f"Lane {int(button - (len(lane_buttons) - 1) / 2)}",
                                                x1=0.7025*frame_width,
                                                y1=((button + 1) / (len(lane_buttons) + 1) + 0.005) * frame_height,
                                                x2=0.9975*frame_width,
                                                y2=((button + 2) / (len(lane_buttons) + 1)  - 0.005) * frame_height,
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

    for lane in range(len(lane_buttons)):
        if lane_buttons[lane] == 1:
            for i in range(2):
                for j in range(len(lanes[lane][i])):
                    x, y, tied_to_edge = lanes[lane][i][j]
                    radius = round(window_height/100)
                    radius = 1 if radius < 1 else radius
                    if grabbed_point != (None, None, None):
                        point_grabbed = True if (lane, i, j) == grabbed_point and left_clicked == True else False
                    else:
                        point_grabbed = True if x * image.shape[1] - radius < mouse_x_image * image.shape[1] < x * image.shape[1] + radius and y * image.shape[0] - radius < mouse_y_image * image.shape[0] < y * image.shape[0] + radius and left_clicked == True else False
                    if left_clicked == False:
                        point_grabbed = False
                        grabbed_point = None, None, None
                    if point_grabbed == True:
                        grabbed_point = lane, i, j
                        if tied_to_edge == False:
                            x = min(1, max(0, mouse_x_image))
                            y = min(1, max(0, mouse_y_image))
                            lanes[lane][i][j] = x, y, tied_to_edge
                        else:
                            nearest_x = 0 if mouse_x_image < 0.5 else 1
                            nearest_y = 0 if mouse_y_image < 0.5 else 1
                            if abs(mouse_x_image - nearest_x) * image.shape[1] < abs(mouse_y_image - nearest_y) * image.shape[0]:
                                x = nearest_x
                                y = mouse_y_image
                            else:
                                x = mouse_x_image
                                y = nearest_y
                            x = min(1, max(0, x))
                            y = min(1, max(0, y))
                            lanes[lane][i][j] = x, y, tied_to_edge
                        cv2.circle(image, (round(x * image.shape[1]), round(y * image.shape[0])), radius, (220, 220, 220), -1, cv2.LINE_AA)
                        allow_adding_points = False
                    elif x * image.shape[1] - radius <= mouse_x_image * image.shape[1] <= x * image.shape[1] + radius and y * image.shape[0] - radius <= mouse_y_image * image.shape[0] <= y * image.shape[0] + radius:
                        cv2.circle(image, (round(x * image.shape[1]), round(y * image.shape[0])), radius, (20, 220, 220), -1, cv2.LINE_AA)
                        allow_adding_points = False
                        if tied_to_edge == False and right_clicked == True and last_right_clicked == False and len(lanes[lane][i]) > 2:
                            remove_list.append((lane, i, j))
                    else:
                        cv2.circle(image, (round(x * image.shape[1]), round(y * image.shape[0])), round(radius * 0.7) if round(radius * 0.7) > 1 else 1, (0, 200, 200), -1, cv2.LINE_AA)

                for j in range(len(lanes[lane][i])):
                    if j == 0:
                        continue
                    x1, y1, _ = lanes[lane][i][j]
                    x2, y2, _ = lanes[lane][i][j - 1]

                    line_vec_x = x2 - x1
                    line_vec_y = y2 - y1
                    line_length = math.sqrt(line_vec_x**2 + line_vec_y**2)
                    if line_length == 0:
                        line_length = 0.0001
                    proj_x = ((mouse_x_image - x1) * line_vec_x + (mouse_y_image - y1) * line_vec_y) / line_length
                    proj_x = max(0, min(1, proj_x / line_length))
                    closest_x = x1 + proj_x * line_vec_x
                    closest_y = y1 + proj_x * line_vec_y
                    distance_to_line = math.sqrt((mouse_x_image - closest_x)**2 + (mouse_y_image - closest_y)**2)

                    if distance_to_line < 0.005:
                        cv2.line(image, (round(x1 * image.shape[1]), round(y1 * image.shape[0])), (round(x2 * image.shape[1]), round(y2 * image.shape[0])), (0, 200, 200), round(window_height/400) if round(window_height/400) > 1 else 1, cv2.LINE_AA)
                        if left_clicked == True and last_left_clicked == False and allow_adding_points == True:
                            allow_adding_points = False
                            insert_index = j
                            lanes[lane][i].insert(insert_index, (closest_x, closest_y, False))
                    else:
                        cv2.line(image, (round(x1 * image.shape[1]), round(y1 * image.shape[0])), (round(x2 * image.shape[1]), round(y2 * image.shape[0])), (0, 200, 200), round(window_height/600) if round(window_height/600) > 1 else 1, cv2.LINE_AA)

                lane_name = f"{int(lane - (len(lane_buttons) - 1) / 2)}{'L' if i == 0 else 'R'}"
                text, fontscale, thickness, width, height = GetTextSize(lane_name, 0.1 * image.shape[1], 0.015 * image.shape[0])
                cv2.putText(image, lane_name, (round(x * image.shape[1] - (width * 1.75 if i == 1 else width * -0.75)), round(y * image.shape[0] - height * 1.5 * (y - 0.75))), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 200, 200), thickness, cv2.LINE_AA)

    if len(remove_list) > 0:
        for lane, i, j in sorted(remove_list, reverse=True):
            if i < len(lanes[lane]) and j < len(lanes[lane][i]):
                lanes[lane][i].pop(j)
                if len(lanes[lane][i]) == 0:
                    lanes[lane].pop(i)
        remove_list.clear()

    frame[0:background.shape[0], 0:round(background.shape[1] * 0.7)] = image

    last_left_clicked = left_clicked
    last_right_clicked = right_clicked
    allow_adding_points = True

    cv2.imshow("LaneDetection - Annotation", frame)
    cv2.waitKey(1)

    time_to_sleep = 1/fps - (time.time() - start)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)