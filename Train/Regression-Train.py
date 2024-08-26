import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from PIL import Image
import numpy as np
import threading
import random
import shutil
import torch
import time
import cv2

# Constants
PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = PATH + "\\Datasets\\FinalDataset"
MODEL_PATH = PATH + "\\Models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10000
BATCH_SIZE = 8
IMG_WIDTH = 400
IMG_HEIGHT = 400
LEARNING_RATE = 0.00001
MAX_LEARNING_RATE = 0.001
TRAIN_VAL_RATIO = 1
NUM_WORKERS = 0
DROPOUT = 0.8
PATIENCE = 500
SHUFFLE = True
PIN_MEMORY = False
DROP_LAST = False
CACHE = True

IMG_COUNT = 0
for file in os.listdir(DATA_PATH):
    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and "#" not in file:
        IMG_COUNT += 1
if IMG_COUNT == 0:
    print("No images found, exiting...")
    exit()

LANES = None
for file in os.listdir(DATA_PATH):
    if "#" not in file:
        count = 0
        while True:
            if os.path.exists(os.path.join(DATA_PATH, file.split('.')[0] + "#" + str(count) + "." + file.split('.')[-1])) == False:
                count -= 1
                break
            count += 1
        LANES = count * 2 + 1
if LANES is None:
    print("No lanes found, exiting...")
    exit()

RED = "\033[91m"
GREEN = "\033[92m"
DARK_GREY = "\033[90m"
NORMAL = "\033[0m"
def timestamp():
    return DARK_GREY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + NORMAL

print("\n----------------------------------------------\n")

print(timestamp() + f"Using {str(DEVICE).upper()} for training")
print(timestamp() + 'Number of CPU cores:', multiprocessing.cpu_count())
print()
print(timestamp() + "Training settings:")
print(timestamp() + "> Epochs:", NUM_EPOCHS)
print(timestamp() + "> Batch size:", BATCH_SIZE)
print(timestamp() + "> Lanes:", LANES)
print(timestamp() + "> Images:", IMG_COUNT)
print(timestamp() + "> Image width:", IMG_WIDTH)
print(timestamp() + "> Image height:", IMG_HEIGHT)
print(timestamp() + "> Learning rate:", LEARNING_RATE)
print(timestamp() + "> Max learning rate:", MAX_LEARNING_RATE)
print(timestamp() + "> Dataset split:", TRAIN_VAL_RATIO)
print(timestamp() + "> Number of workers:", NUM_WORKERS)
print(timestamp() + "> Dropout:", DROPOUT)
print(timestamp() + "> Patience:", PATIENCE)
print(timestamp() + "> Shuffle:", SHUFFLE)
print(timestamp() + "> Pin memory:", PIN_MEMORY)
print(timestamp() + "> Drop last:", DROP_LAST)
print(timestamp() + "> Cache:", CACHE)


class custom():
    class RandomHorizontalFlip():
        pass

# Custom dataset class
if CACHE:
    def load_data(files=None, type=None):
        images = []
        labels = []
        print(f"\r{timestamp()}Caching {type} dataset...           ", end='', flush=True)
        for file in os.listdir(DATA_PATH):
            if file in files:
                img = cv2.imread(os.path.join(DATA_PATH, file), cv2.IMREAD_UNCHANGED)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img / 255.0

                lanes = []
                for i in range(LANES):
                    label_img = cv2.imread(os.path.join(DATA_PATH, file.split('.')[0] + "#" + str(i - LANES // 2) + "." + file.split('.')[-1]), cv2.IMREAD_UNCHANGED)
                    if len(label_img.shape) == 3:
                        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
                    label_img = cv2.resize(label_img, (IMG_WIDTH, IMG_HEIGHT))
                    label_img = label_img / 255.0
                    lanes.append(label_img)
                lanes.reverse()

                images.append(img)
                labels.append(lanes)

            if len(images) % round(len(files) / 100) if round(len(files) / 100) != 0 else 1 == 0:
                print(f"\r{timestamp()}Caching {type} dataset... ({round(100 * len(images) / len(files))}%)", end='', flush=True)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx].copy()
            label = self.labels[idx].copy()
            for transform in self.transform:
                if isinstance(transform, custom.RandomHorizontalFlip):
                    if random.uniform(0, 1) < 0.5:
                        image = cv2.flip(image, 1)
                        for i, img in enumerate(label):
                            label[i] = cv2.flip(img, 1)
                else:
                    image = transform(image)
                    for img in label:
                        img = transform(img)
            return image, torch.as_tensor(label, dtype=torch.float32)

else:

    class CustomDataset(Dataset):
        def __init__(self, files=None, transform=None):
            self.files = files
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            file = self.files[index]

            img = cv2.imread(os.path.join(DATA_PATH, file), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0

            lanes = []
            for i in range(LANES):
                label_img = cv2.imread(os.path.join(DATA_PATH, file.split('.')[0] + "#" + str(i - LANES // 2) + "." + file.split('.')[-1]), cv2.IMREAD_UNCHANGED)
                if len(label_img.shape) == 3:
                    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
                label_img = cv2.resize(label_img, (IMG_WIDTH, IMG_HEIGHT))
                label_img = label_img / 255.0
                lanes.append(label_img)
            lanes.reverse()
            label = np.array(lanes, dtype=np.float32)

            image = np.array(img, dtype=np.float32)
            for transform in self.transform:
                if isinstance(transform, custom.RandomHorizontalFlip):
                    if random.uniform(0, 1) < 0.5:
                        image = cv2.flip(image, 1)
                        for i, img in enumerate(label):
                            label[i] = cv2.flip(img, 1)
                else:
                    image = transform(image)
                    for img in label:
                        img = transform(img)
            return image, torch.as_tensor(label, dtype=torch.float32)

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 256, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv9 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv10 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, 1, 3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        
        # Decoder
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.up1(x)
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.conv11(x)
        x = self.up2(x)
        
        x = self.activation(x)
        
        return x

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

def generate_tensorboard_image(model, dataset, resolution):
    random_index = random.randint(0, len(dataset) - 1)
    image, label = dataset[random_index]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(DEVICE))

    image = cv2.resize(cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2BGRA), (resolution, resolution))
    frame = np.zeros((resolution, round(resolution * 1.5), 4), dtype=np.float32)
    frame[0:image.shape[0], 0:image.shape[1]] = image

    for i, label_img in enumerate(label):
        label_img = cv2.resize(cv2.cvtColor(label_img.numpy(), cv2.COLOR_GRAY2BGRA), (resolution, resolution))
        label_img[:, :, 0] = 0
        label_img[:, :, 2] = 0
        frame[0:image.shape[0], 0:image.shape[1]] = cv2.addWeighted(frame[0:image.shape[0], 0:image.shape[1]], 1, label_img, 0.2, 0)

    prediction = prediction.squeeze(0).cpu()
    for i, pred_img in enumerate(prediction):
        pred_img = cv2.resize(cv2.cvtColor(pred_img.numpy(), cv2.COLOR_GRAY2BGRA), (resolution, resolution))
        label_img[:, :, 1] = 0
        pred_img[:, :, 2] = 0
        frame[0:image.shape[0], 0:image.shape[1]] = cv2.addWeighted(frame[0:image.shape[0], 0:image.shape[1]], 1, pred_img, 0.5, 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("image", frame)
    cv2.waitKey(1)
    return frame

def main():
    # Initialize model
    model = NeuralNetwork().to(DEVICE)

    def get_model_size_mb(model):
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.size())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        bytes_per_param = next(model.parameters()).element_size()
        model_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
        return total_params, trainable_params, non_trainable_params, model_size_mb

    total_params, trainable_params, non_trainable_params, model_size_mb = get_model_size_mb(model)

    print()
    print(timestamp() + "Model properties:")
    print(timestamp() + f"> Total parameters: {total_params}")
    print(timestamp() + f"> Trainable parameters: {trainable_params}")
    print(timestamp() + f"> Non-trainable parameters: {non_trainable_params}")
    print(timestamp() + f"> Predicted model size: {model_size_mb:.2f}MB")

    print("\n----------------------------------------------\n")

    print(timestamp() + "Loading...")

    # Create tensorboard logs folder if it doesn't exist
    if not os.path.exists(f"{PATH}/Train/logs"):
        os.makedirs(f"{PATH}/Train/logs")

    # Delete previous tensorboard logs
    for obj in os.listdir(f"{PATH}/Train/logs"):
        try:
            shutil.rmtree(f"{PATH}/Train/logs/{obj}")
        except:
            os.remove(f"{PATH}/Train/logs/{obj}")

    # Tensorboard setup
    summary_writer = SummaryWriter(f"{PATH}/Train/logs", comment="Regression-Training", flush_secs=20)

    # Transformations
    train_transform = (
        custom.RandomHorizontalFlip(),
        transforms.ToTensor()
    )

    val_transform = (
        custom.RandomHorizontalFlip(),
        transforms.ToTensor()
    )

    # Create datasets
    all_files = [f for f in os.listdir(DATA_PATH) if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")) and "#" not in f]
    random.shuffle(all_files)
    train_size = int(len(all_files) * TRAIN_VAL_RATIO)
    val_size = len(all_files) - train_size
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    if train_size == 0 or val_size == 0:
        if len(train_files) > len(val_files):
            val_files = train_files
        else:
            train_files = val_files

    if CACHE:
        train_images, train_labels = load_data(train_files, "train")
        val_images, val_labels = load_data(val_files, "val")
        train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
        val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)
    else:
        train_dataset = CustomDataset(train_files, transform=train_transform)
        val_dataset = CustomDataset(val_files, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

    # Initialize scaler, loss function, optimizer and scheduler
    scaler = GradScaler()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)

    # Early stopping variables
    best_validation_loss = float('inf')
    best_model = None
    best_model_epoch = None
    best_model_training_loss = None
    best_model_validation_loss = None
    wait = 0

    print(f"\r{timestamp()}Starting training...                ")
    print("\n-----------------------------------------------------------------------------------------------------------\n")

    training_time_prediction = time.time()
    training_start_time = time.time()
    epoch_total_time = 0
    training_loss = 0
    validation_loss = 0
    training_epoch = 0

    global PROGRESS_PRINT
    PROGRESS_PRINT = "initializing"
    def training_progress_print():
        global PROGRESS_PRINT
        def num_to_str(num: int):
            str_num = format(num, '.15f')
            while len(str_num) > 15:
                str_num = str_num[:-1]
            while len(str_num) < 15:
                str_num = str_num + '0'
            return str_num
        while PROGRESS_PRINT == "initializing":
            time.sleep(1)
        last_message = ""
        while PROGRESS_PRINT == "running":
            progress = (time.time() - epoch_total_start_time) / epoch_total_time
            if progress > 1: progress = 1
            if progress < 0: progress = 0
            progress = '█' * round(progress * 10) + '░' * (10 - round(progress * 10))
            epoch_time = round(epoch_total_time, 2) if epoch_total_time > 1 else round((epoch_total_time) * 1000)
            eta = time.strftime('%H:%M:%S', time.gmtime(round((training_time_prediction - training_start_time) / (training_epoch) * NUM_EPOCHS - (training_time_prediction - training_start_time) + (training_time_prediction - time.time()), 2)))
            message = f"{progress} Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}, {epoch_time}{'s' if epoch_total_time > 1 else 'ms'}/Epoch, ETA: {eta}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
            last_message = message
            time.sleep(1)
        if PROGRESS_PRINT == "early stopped":
            message = f"Early stopping at Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
        elif PROGRESS_PRINT == "finished":
            message = f"Finished at Epoch {training_epoch}, Train Loss: {num_to_str(training_loss)}, Val Loss: {num_to_str(validation_loss)}"
            print(f"\r{message}" + (" " * (len(last_message) - len(message)) if len(last_message) > len(message) else ""), end='', flush=True)
        PROGRESS_PRINT = "received"
    threading.Thread(target=training_progress_print, daemon=True).start()

    for epoch, _ in enumerate(range(NUM_EPOCHS), 1):
        epoch_total_start_time = time.time()


        epoch_training_start_time = time.time()

        # Training phase
        model.train()
        running_training_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_training_loss += loss.item()
        running_training_loss /= len(train_dataloader)
        training_loss = running_training_loss

        epoch_training_time = time.time() - epoch_training_start_time


        epoch_validation_start_time = time.time()

        # Validation phase
        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad(), autocast():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data[0].to(DEVICE, non_blocking=True), data[1].to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_validation_loss += loss.item()
        running_validation_loss /= len(val_dataloader)
        validation_loss = running_validation_loss

        epoch_validation_time = time.time() - epoch_validation_start_time


        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model
            best_model_epoch = epoch
            best_model_training_loss = training_loss
            best_model_validation_loss = validation_loss
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE and PATIENCE > 0:
                epoch_total_time = time.time() - epoch_total_start_time
                # Log values to Tensorboard
                summary_writer.add_scalars("Stats", {
                    'train_loss': training_loss,
                    'validation_loss': validation_loss,
                    'epoch_total_time': epoch_total_time,
                    'epoch_training_time': epoch_training_time,
                    'epoch_validation_time': epoch_validation_time
                }, epoch)
                summary_writer.add_image("Image", generate_tensorboard_image(model, val_dataset, 700), global_step=epoch, dataformats="HWC")
                training_time_prediction = time.time()
                PROGRESS_PRINT = "early stopped"
                break

        epoch_total_time = time.time() - epoch_total_start_time

        # Log values to Tensorboard
        summary_writer.add_scalars(f'Stats', {
            'train_loss': training_loss,
            'validation_loss': validation_loss,
            'epoch_total_time': epoch_total_time,
            'epoch_training_time': epoch_training_time,
            'epoch_validation_time': epoch_validation_time
        }, epoch)
        summary_writer.add_image("Image", generate_tensorboard_image(model, val_dataset, 700), global_step=epoch, dataformats="HWC")
        training_epoch = epoch
        training_time_prediction = time.time()
        PROGRESS_PRINT = "running"

    if PROGRESS_PRINT != "early stopped":
        PROGRESS_PRINT = "finished"
    while PROGRESS_PRINT != "received":
        time.sleep(1)

    print("\n\n-----------------------------------------------------------------------------------------------------------")

    TRAINING_TIME = time.strftime('%H-%M-%S', time.gmtime(time.time() - training_start_time))
    TRAINING_DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    print()
    print(timestamp() + f"Training completed after " + TRAINING_TIME.replace('-', ':'))

    # Save the last model
    print(timestamp() + "Saving the last model...")

    torch.cuda.empty_cache()

    model.eval()
    total_train = 0
    correct_train = 0
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()
    training_dataset_accuracy = str(round(100 * (correct_train / total_train), 2)) + "%"

    torch.cuda.empty_cache()

    total_val = 0
    correct_val = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()
    validation_dataset_accuracy = str(round(100 * (correct_val / total_val), 2)) + "%"

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(criterion).replace('\n', '')
    metadata_model = str(model).replace('\n', '')
    metadata = (f"epochs#{epoch}",
                f"batch#{BATCH_SIZE}",
                f"lanes#{LANES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_WIDTH}",
                f"image_height#{IMG_HEIGHT}",
                f"learning_rate#{LEARNING_RATE}",
                f"max_learning_rate#{MAX_LEARNING_RATE}",
                f"dataset_split#{TRAIN_VAL_RATIO}",
                f"number_of_workers#{NUM_WORKERS}",
                f"dropout#{DROPOUT}",
                f"patience#{PATIENCE}",
                f"shuffle#{SHUFFLE}",
                f"pin_memory#{PIN_MEMORY}",
                f"training_time#{TRAINING_TIME}",
                f"training_date#{TRAINING_DATE}",
                f"training_device#{DEVICE}",
                f"training_os#{os.name}",
                f"architecture#{metadata_model}",
                f"torch_version#{torch.__version__}",
                f"numpy_version#{np.__version__}",
                f"pil_version#{Image.__version__}",
                f"train_transform#{train_transform}",
                f"val_transform#{val_transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{best_model_training_loss}",
                f"validation_loss#{best_model_validation_loss}",
                f"training_dataset_accuracy#{training_dataset_accuracy}",
                f"validation_dataset_accuracy#{validation_dataset_accuracy}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    last_model_saved = False
    for i in range(5):
        try:
            last_model = torch.jit.script(model)
            torch.jit.save(last_model, os.path.join(MODEL_PATH, f"RegressionModel-LAST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            last_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the last model. Retrying...")
    print(timestamp() + "Last model saved successfully.") if last_model_saved else print(timestamp() + "Failed to save the last model.")

    # Save the best model
    print(timestamp() + "Saving the best model...")

    torch.cuda.empty_cache()

    best_model.eval()
    total_train = 0
    correct_train = 0
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()
    training_dataset_accuracy = str(round(100 * (correct_train / total_train), 2)) + "%"

    torch.cuda.empty_cache()

    total_val = 0
    correct_val = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()
    validation_dataset_accuracy = str(round(100 * (correct_val / total_val), 2)) + "%"

    metadata_optimizer = str(optimizer).replace('\n', '')
    metadata_criterion = str(criterion).replace('\n', '')
    metadata_model = str(best_model).replace('\n', '')
    metadata = (f"epochs#{best_model_epoch}",
                f"batch#{BATCH_SIZE}",
                f"lanes#{LANES}",
                f"image_count#{IMG_COUNT}",
                f"image_width#{IMG_WIDTH}",
                f"image_height#{IMG_HEIGHT}",
                f"learning_rate#{LEARNING_RATE}",
                f"max_learning_rate#{MAX_LEARNING_RATE}",
                f"dataset_split#{TRAIN_VAL_RATIO}",
                f"number_of_workers#{NUM_WORKERS}",
                f"dropout#{DROPOUT}",
                f"patience#{PATIENCE}",
                f"shuffle#{SHUFFLE}",
                f"pin_memory#{PIN_MEMORY}",
                f"training_time#{TRAINING_TIME}",
                f"training_date#{TRAINING_DATE}",
                f"training_device#{DEVICE}",
                f"training_os#{os.name}",
                f"architecture#{metadata_model}",
                f"torch_version#{torch.__version__}",
                f"numpy_version#{np.__version__}",
                f"pil_version#{Image.__version__}",
                f"train_transform#{train_transform}",
                f"val_transform#{val_transform}",
                f"optimizer#{metadata_optimizer}",
                f"loss_function#{metadata_criterion}",
                f"training_size#{train_size}",
                f"validation_size#{val_size}",
                f"training_loss#{training_loss}",
                f"validation_loss#{validation_loss}",
                f"training_dataset_accuracy#{training_dataset_accuracy}",
                f"validation_dataset_accuracy#{validation_dataset_accuracy}")
    metadata = {"data": metadata}
    metadata = {data: str(value).encode("ascii") for data, value in metadata.items()}

    best_model_saved = False
    for i in range(5):
        try:
            best_model = torch.jit.script(best_model)
            torch.jit.save(best_model, os.path.join(MODEL_PATH, f"RegressionModel-BEST-{TRAINING_DATE}.pt"), _extra_files=metadata)
            best_model_saved = True
            break
        except:
            print(timestamp() + "Failed to save the best model. Retrying...")
    print(timestamp() + "Best model saved successfully.") if best_model_saved else print(timestamp() + "Failed to save the best model.")

    print("\n----------------------------------------------\n")

if __name__ == '__main__':
    main()