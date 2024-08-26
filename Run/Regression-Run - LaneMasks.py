from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import bettercam
import torch
import time
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = bettercam.create(output_color="BGR", output_idx=0)

PATH = os.path.dirname(os.path.dirname(__file__)) + "\\Models"
MODEL_PATH = ""
for file in os.listdir(PATH):
    if file.endswith(".pt") and "LaneMasks" in file:
        MODEL_PATH = os.path.join(PATH, file)
        break
if MODEL_PATH == "":
    print("No model found.")
    exit()

print(f"\nModel: {MODEL_PATH}")

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=device)
model.eval()

metadata = eval(metadata["data"])
for var in metadata:
    if "lanes" in var:
        LANES = int(var.split("#")[1])
    if "image_width" in var:
        IMG_WIDTH = int(var.split("#")[1])
    if "image_height" in var:
        IMG_HEIGHT = int(var.split("#")[1])
    if "image_channels" in var:
        IMG_CHANNELS = str(var.split("#")[1])
    if "training_dataset_accuracy" in var:
        print("Training dataset accuracy: " + str(var.split("#")[1]))
    if "validation_dataset_accuracy" in var:
        print("Validation dataset accuracy: " + str(var.split("#")[1]))

cv2.namedWindow("LaneDetection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("LaneDetection", cv2.WND_PROP_TOPMOST, 1)

def get_text_size(text="NONE", text_width=100, max_text_height=100):
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

def generate_image(model, frame, resolution):
    with torch.no_grad():
        prediction = model(frame.unsqueeze(0).to(device))

    frame = cv2.resize(cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2BGRA), (resolution, resolution))

    prediction = prediction.squeeze(0).cpu()
    for i, pred_img in enumerate(prediction):
        pred_img = cv2.resize(cv2.cvtColor(pred_img.numpy(), cv2.COLOR_GRAY2BGRA), (resolution, resolution))
        pred_img[:, :, 2] = 0
        frame[0:frame.shape[0], 0:frame.shape[1]] = cv2.addWeighted(frame[0:frame.shape[0], 0:frame.shape[1]], 1, pred_img, 0.5, 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

transform = transforms.Compose([
    transforms.ToTensor(),
])

while True:
    start = time.time()
    frame = camera.grab()
    if frame is None:
        continue

    frame = frame[round(frame.shape[0] * 0.52):, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    frame = transform(frame)

    frame = generate_image(model, frame, resolution=700)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow("LaneDetection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()