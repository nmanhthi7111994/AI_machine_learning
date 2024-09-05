import os
import random
import shutil

# Define base paths
base_train_path = 'train/images'
base_validate_path = 'validate/images'
base_test_path = 'test/images'
base_train_labels_path = 'train/labels'
base_validate_labels_path = 'validate/labels'
base_test_labels_path = 'test/labels'

# Set the proportion of the dataset to use for validation and testing
validate_ratio = 0.1  # 10% for validation
test_ratio = 0.1      # 10% for testing

# Function to create necessary directories
def create_directories(base_path, class_names):
    for class_name in class_names:
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

# Function to move files
def move_files(file_list, src_image_path, src_label_path, dest_image_path, dest_label_path, class_name):
    for image_file in file_list:
        # Construct full file paths for images and labels
        src_img = os.path.join(src_image_path, class_name, image_file)
        src_lbl = os.path.join(src_label_path, class_name, image_file.replace(os.path.splitext(image_file)[1], '.txt'))
        dest_img = os.path.join(dest_image_path, class_name, image_file)
        dest_lbl = os.path.join(dest_label_path, class_name, os.path.basename(src_lbl))

        # Move image if exists
        if os.path.exists(src_img):
            shutil.move(src_img, dest_img)
        
        # Move label if exists
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dest_lbl)

# Get all class folders
class_folders = [d for d in os.listdir(base_train_path) if os.path.isdir(os.path.join(base_train_path, d))]

# Create directories for validation and test sets
create_directories(base_validate_path, class_folders)
create_directories(base_test_path, class_folders)
create_directories(base_validate_labels_path, class_folders)
create_directories(base_test_labels_path, class_folders)

# Iterate over each class folder
for class_name in class_folders:
    # Path to the current class folder in the train set
    class_image_path = os.path.join(base_train_path, class_name)
    class_label_path = os.path.join(base_train_labels_path, class_name)

    # List all image files in the current class folder
    all_images = os.listdir(class_image_path)

    # Filter to only include image files
    all_images = [f for f in all_images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Shuffle images to ensure random selection
    random.shuffle(all_images)

    # Calculate the number of images for validation and testing
    num_images = len(all_images)
    num_validate = int(num_images * validate_ratio)
    num_test = int(num_images * test_ratio)

    # Split the dataset for the current class
    validate_images = all_images[:num_validate]
    test_images = all_images[num_validate:num_validate + num_test]

    # Move validation images and labels
    move_files(validate_images, base_train_path, base_train_labels_path, base_validate_path, base_validate_labels_path, class_name)

    # Move test images and labels
    move_files(test_images, base_train_path, base_train_labels_path, base_test_path, base_test_labels_path, class_name)

    print(f"Moved {len(validate_images)} images to validation set for class {class_name}.")
    print(f"Moved {len(test_images)} images to test set for class {class_name}.")
