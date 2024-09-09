import os
import shutil

# Define paths
image_folder = '/application/AI/train/images'      # Folder containing images
label_folder = '/application/AI/train/labels'      # Folder containing label files
output_image_folder = '/application/AI/validate/images'  # Folder to save filtered images
output_label_folder = '/application/AI/validate/labels'  # Folder to save filtered labels

# Ensure the output folders exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Define target classes
target_classes = {'6','7','8'}

# Function to check if any class in a label file matches target classes
def contains_target_class(label_path, target_classes):
    with open(label_path, 'r') as file:
        for line in file:
            class_id = line.split()[0]
            if class_id in target_classes:
                return True
    return False

# Iterate through label files
for label_file in os.listdir(label_folder):
    if label_file.endswith('.txt'):  # Ensure you're reading only label files
        label_path = os.path.join(label_folder, label_file)
        
        # Check if the label contains target classes
        if contains_target_class(label_path, target_classes):
            # Corresponding image file (assuming the same name but different extension)
            image_file = label_file.replace('.txt', '.jpg')  # Adjust extension if necessary
            image_path = os.path.join(image_folder, image_file)
            
            # Check if the image exists before copying
            if os.path.exists(image_path):
                # Copy the image to the output folder
                shutil.copy(image_path, os.path.join(output_image_folder, image_file))
                print(f"Copied Image: {image_file}")
                
                # Copy the label to the output folder
                shutil.copy(label_path, os.path.join(output_label_folder, label_file))
                print(f"Copied Label: {label_file}")

print("Filtering and copying of images and labels completed.")
