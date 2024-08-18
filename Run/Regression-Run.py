from torchvision import transforms
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
    if "classes" in var:
        CLASSES = int(var.split("#")[1])
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
    if "val_transform" in var:
        transform = var.replace("\\n", "\n").replace('\\', '').split("#")[1]
        transform_list = []
        transform_parts = transform.strip().split("\n")
        for part in transform_parts[1:-1]:
            part = part.strip()
            if part:
                try:
                    transform_args = []
                    transform_name = part.split("(")[0]
                    if "(" in part:
                        args = part.split("(")[1][:-1].split(",")
                        for arg in args:
                            try:
                                transform_args.append(int(arg.strip()))
                            except ValueError:
                                try:
                                    transform_args.append(float(arg.strip()))
                                except ValueError:
                                    transform_args.append(arg.strip())
                    if transform_name == "ToTensor":
                        transform_list.append(transforms.ToTensor())
                    else:
                        transform_list.append(getattr(transforms, transform_name)(*transform_args))
                except (AttributeError, IndexError, ValueError):
                    print(f"Skipping or failed to create transform: {part}")
        transform = transforms.Compose(transform_list)

cv2.namedWindow("LaneDetection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("LaneDetection", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("LaneDetection", IMG_WIDTH, IMG_HEIGHT)

while True:
    start = time.time()
    frame = camera.grab()
    if frame is None:
        continue

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = np.array(frame, dtype=np.float32) / 255.0

    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(device))
        output = output.tolist()

    print(output)

    cv2.putText(frame, f"FPS: {round(1 / (time.time() - start), 1)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("LaneDetection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()