import os
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image

# Function to calculate SSIM between two images
def calculate_ssim(image1_path, image2_path):
    # Load the images and convert them to grayscale
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to a common size for comparison
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))
    
    # Compute the SSIM between the two images
    score, _ = ssim(image1, image2, full=True)
    return score

# Path to your dataset directory
dataset_path = '/content/test/images'

# Store paths of images already processed
processed_images = []
duplicates = []

# Iterate over images in the dataset directory
for root, _, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        # Skip non-image files
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        # Compare the current image with all previously processed images
        for processed_image_path in processed_images:
            similarity_score = calculate_ssim(file_path, processed_image_path)
            
            # Check if the images match with at least 95% similarity
            if similarity_score >= 0.80:
                print(f"Duplicate found with 95% similarity: {file_path} and {processed_image_path}")
                duplicates.append(file_path)
                break  # Skip further comparisons once a duplicate is found

        # Add the current image to the processed list
        processed_images.append(file_path)

# Optional: Remove duplicates (be careful with this step!)
# for duplicate in duplicates:
#     os.remove(duplicate)
