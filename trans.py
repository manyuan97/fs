import os
import shutil

# Define the source and destination directories
source_dirs = ["f_regression_y1_LR_100", "f_regression_y1_LR_200", "f_regression_у1_LR_400",
               "f_regression_у1_LR_500", "f_regression_у1_LR_800", "f_regression_у1_LR_1000"]
destination_dir = "f_regression_cross_LR_y1"

# Iterate over each source directory
for source_dir in source_dirs:
    # Path to source directory
    source_path = os.path.join(os.getcwd(), source_dir)
    # Path to destination directory
    destination_path = os.path.join(os.getcwd(), destination_dir)
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    # List all files in the source directory
    files = os.listdir(source_path)
    
    # Iterate over each file in the source directory
    for file in files:
        # Check if it's a JSON file
        if file.endswith(".json"):
            # Generate new filename with format k_{}_results.json
            new_filename = "k_{}_results.json".format(file.split("_")[3])
            # Move the file to the destination directory with the new filename
            shutil.move(os.path.join(source_path, file), os.path.join(destination_path, new_filename))
