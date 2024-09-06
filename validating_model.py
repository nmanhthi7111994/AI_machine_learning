import cv2
import os
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Import for displaying images in Colab

# Define paths
model_path = '/content/simple_sign_tranning_1.pt'  # Path to your trained model
input_folder = '/content/images'  # Path to the folder containing images
output_folder = 'output'  # Path to save the output images

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(model_path)

# Iterate over all images in the input folder
for image_name in os.listdir(input_folder):
    # Full path to the image
    image_path = os.path.join(input_folder, image_name)
    
    # Check if the file is an image
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    
    # Perform detection
    results = model(image_path)
    
    # Display the results on the image
    for result in results:
        # Result object to image
        detected_img = result.plot()  # This draws bounding boxes and labels on the image
        
        # Show the image with detections
        cv2_imshow(detected_img)  # Use cv2_imshow to display images in Colab
        
        # Save the output image
        output_path = os.path.join(output_folder, f"detected_{image_name}")
        cv2.imwrite(output_path, detected_img)

# Close all OpenCV windows (not needed for Colab but good for local runs)
cv2.destroyAllWindows()
