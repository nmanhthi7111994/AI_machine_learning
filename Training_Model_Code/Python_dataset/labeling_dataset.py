import os
import pandas as pd

# Path to your CSV file
csv_file = '/path/to/your/dataset.csv'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Directory where the label files will be saved
label_dir = 'labels/'
os.makedirs(label_dir, exist_ok=True)

# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo_format(width, height, x1, y1, x2, y2):
    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height
    return x_center, y_center, bbox_width, bbox_height

# Process each row in the dataframe
for index, row in df.iterrows():
    width = row['Width']
    height = row['Height']
    x1 = row['Roi.X1']
    y1 = row['Roi.Y1']
    x2 = row['Roi.X2']
    y2 = row['Roi.Y2']
    class_id = row['ClassId']
    image_path = row['Path']

    # Convert bounding box to YOLO format
    x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(width, height, x1, y1, x2, y2)

    # Generate label file path
    label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

    # Write label data to file
    with open(label_path, 'w') as label_file:
        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("Label files created successfully from CSV data.")
