import cv2
import os
import pandas as pd
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Import for displaying images in Colab

# Define paths
model_path = '/content/best_0827.pt'  # Path to your trained model
input_folder = '/content/validate/images'           # Path to the folder containing images
label_folder = '/content/validate/labels'           # Path to the folder containing ground truth labels
output_folder = '/content/output'          # Path to save the output images
csv_output_path = '/content/evaluation_results.csv'  # Path to save the evaluation results CSV

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(model_path)

# Initialize lists to store evaluation results
results_data = []

# Function to parse label file into a list of ground truth classes
def parse_label_file(label_path):
    classes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = parts[0]
            # Map class IDs to class names if necessary
            class_name = {
                '5': 'SpeedLimit_50',
                '6': 'SpeedLimit_60',
                '7': 'SpeedLimit_70'
            }.get(class_id, class_id)  # Default to class_id if not found
            classes.append(class_name)
    return classes

# Iterate over all images in the input folder
for image_name in os.listdir(input_folder):
    # Full path to the image
    image_path = os.path.join(input_folder, image_name)

    # Check if the file is an image
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    # Perform detection
    results = model(image_path)

    # Extract detected class names
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

    # Corresponding label file
    label_file = image_name.replace('.jpg', '.txt')  # Adjust extension if necessary
    label_path = os.path.join(label_folder, label_file)

    # Parse ground truth classes from label file
    if os.path.exists(label_path):
        gt_classes = parse_label_file(label_path)
    else:
        gt_classes = []

    # Compare detections with ground truth classes
    tp = len(set(detected_classes) & set(gt_classes))  # True Positives: correct class detections
    fp = len(detected_classes) - tp  # False Positives: incorrect class detections
    fn = len(gt_classes) - tp  # False Negatives: missed ground truth classes

    # Save results
    results_data.append({
        'Image': image_name,
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'Detected Classes': detected_classes,
        'Ground Truth Classes': gt_classes
    })

    # Save the output image with detections
    detected_img = results[0].plot()
    output_path = os.path.join(output_folder, f"detected_{image_name}")
    cv2.imwrite(output_path, detected_img)
    cv2_imshow(detected_img)  # Display the image in Colab

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(results_data)
df['Precision'] = df['True Positives'] / (df['True Positives'] + df['False Positives'])
df['Recall'] = df['True Positives'] / (df['True Positives'] + df['False Negatives'])
df['F1 Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])

# Save evaluation results to CSV
df.to_csv(csv_output_path, index=False)

print("Evaluation completed and results saved to:", csv_output_path)
