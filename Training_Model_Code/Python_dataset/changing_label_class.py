import os

# Define the path to your labels folder
labels_folder = '/application/AI/Self_trainning/train/labels/70kmh/'
class_id = "2"

# Iterate over each file in the labels folder
for filename in os.listdir(labels_folder):
    if filename.endswith('.txt'):  # Ensure we are only processing text files
        file_path = os.path.join(labels_folder, filename)

        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Change all class IDs to '2'
        updated_lines = []
        for line in lines:
            line_parts = line.strip().split()
            if line_parts:  # Ensure the line is not empty
                line_parts[0] = '2'  # Change the class ID to '2'
            updated_lines.append(' '.join(line_parts) + '\n')

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

print(f"All class IDs changed to {class_id}.")
