# Train the model
#!yolo task=detect mode=train model=/content/best.pt data=/content/Self_trainning/data.yaml epochs=20 imgsz=640 batch=32

import os
import glob
from google.colab import files

# Define the base path where YOLO saves the training results
base_path = "/content/ultralytics/runs/detect/"

# Get a list of all training folders
train_folders = glob.glob(os.path.join(base_path, 'train*'))

# Find the most recently modified training folder
latest_train_folder = max(train_folders, key=os.path.getmtime)

# Construct the path to the best model
model_path = os.path.join(latest_train_folder, "weights", "best.pt")

# Check if the model file exists and download it
if os.path.exists(model_path):
    print(f"Training completed. Model found in {latest_train_folder}. Preparing to download...")
    files.download(model_path)
else:
    print("Model not found. Please check the training process and paths.")
