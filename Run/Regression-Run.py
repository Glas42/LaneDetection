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
    if file.endswith(".pt"):
        MODEL_PATH = os.path.join(PATH, file)
        break
if MODEL_PATH == "":
    print("No model found.")
    exit()

print(f"\nModel: {MODEL_PATH}")

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=device)
model.eval()

metadata = str(metadata["data"]).replace('b"(', '').replace(')"', '').replace("'", "").split(", ") # now in the format: ["key#value", "key#value", ...]
for var in metadata:
    if "outputs" in var:
        OUTPUTS = int(var.split("#")[1])
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

def generate_image(model, image, resolution):
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).tolist()[0]
    image = cv2.resize(cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2BGR), (resolution, resolution))
    frame = np.zeros((resolution, round(resolution * 1.5), 3), dtype=np.float32)
    frame[0:image.shape[0], 0:image.shape[1]] = image
    for i, value in enumerate(prediction[0:LANES * 2]):
        if i % 2 == 0:
            value_1 = value
            value_2 = prediction[0:LANES * 2][i + 1]
            text, fontscale, thickness, width, height = get_text_size(f"Lane {i // 2 - 3}: {round(F.softmax(torch.tensor([value_1, value_2]), dim=0).tolist()[0], 3)}", text_width=0.95 * frame.shape[1] - resolution, max_text_height=0.03 * frame.shape[0])
            cv2.putText(frame, text, (round(resolution + height * 0.5), round((i + 1) * height * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), thickness)
    points_per_lane = (OUTPUTS - LANES * 2) / (LANES * 2)
    for i in range(LANES):
        for j in range(2):
            last_point = None
            points = prediction[int(LANES * 2 + points_per_lane * i * 2 + points_per_lane * j):int(LANES * 2 + points_per_lane * (i * 2 + 1) + points_per_lane * j)]
            for k, x in enumerate(points):
                y = (k / (points_per_lane - 1)) ** 3
                if last_point != None:
                    cv2.line(frame, (round(last_point[0] * resolution), round(last_point[1] * resolution)), (round(x * resolution), round(y * resolution)), (0, 255, 255), 2)
                last_point = x, y
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

    avg_color = 70
    mean_color = np.mean(frame)
    if mean_color != avg_color:
        scaling_factor = avg_color / mean_color
        frame = cv2.multiply(frame, scaling_factor)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    frame = cv2.addWeighted(frame, 4, cv2.GaussianBlur(frame, (0, 0), 20), -4, 0)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    frame = transform(frame)

    frame = generate_image(model, frame, resolution=700)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow("LaneDetection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()