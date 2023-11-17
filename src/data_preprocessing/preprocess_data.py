# src/data_preprocessing/preprocess_data.py

import os
from PIL import Image
from tqdm import tqdm  # If not installed, run: pip install tqdm
from pathlib import Path

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess images in the input directory and save the processed images to the output directory.

    Parameters:
    - input_dir (str): Path to the directory containing class folders in the dataset.
    - output_dir (str): Path to the directory where processed images will be saved.
    - target_size (tuple): Target size for resizing images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of class folders
    classes = os.listdir(input_dir)

    # Process and save each image in each class
    for class_name in tqdm(classes, desc="Processing Classes"):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(input_class_dir):
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_class_dir, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                img_resized = img.resize(target_size)

                # Save the resized image
                img_resized.save(output_path)

if __name__ == "__main__":
    # Example usage
    input_directory = "data\\raw"        # Update with the path to your raw data directory
    output_directory = "data\\processed"  # Update with the desired path for processed data

    preprocess_images(input_directory, output_directory)
