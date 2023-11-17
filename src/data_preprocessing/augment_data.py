# src/data_preprocessing/augment_data.py

import os
import shutil
import imgaug.augmenters as iaa
import imgaug as ia
import cv2
from tqdm import tqdm
from PIL import Image

def augment_images(input_dir, output_dir, augment_factor=2):
    """
    Apply data augmentation to images and save the augmented images to the output directory.

    Parameters:
    - input_dir (str): Path to the directory containing class folders in the dataset.
    - output_dir (str): Path to the directory where augmented images will be saved.
    - augment_factor (int): Factor by which to augment the dataset (e.g., 2 means doubling the dataset).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of class folders
    classes = os.listdir(input_dir)

    # Define augmentation sequence
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),      # Horizontal flip
        iaa.Flipud(0.5),       # Vertical flip
        iaa.Affine(rotate=(-10, 10)),  # Rotation
        iaa.GaussianBlur(sigma=(0, 1.0))  # Gaussian blur
        # Add more augmentation techniques as needed
    ])

    # Apply augmentation and save each image
    for class_name in tqdm(classes, desc="Augmenting Classes"):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # List all image files in the class directory
        image_files = [f for f in os.listdir(input_class_dir) if f.endswith(('.JPG', '.JPEG', '.PNG'))]

        # Augment and save images
        for filename in image_files:
            input_path = os.path.join(input_class_dir, filename)

            # Read the image
            img = ia.imresize_single_image(cv2.imread(input_path), (256,256))

            # Augment the image multiple times
            for i in range(augment_factor):
                augmented_img = augmenter.augment_image(img)
                output_path = os.path.join(output_class_dir, f"{filename.split('.')[0]}_aug_{i}.{filename.split('.')[-1]}")
                Image.fromarray(augmented_img).save(output_path)

if __name__ == "__main__":
    # Example usage
    input_directory = "data\\processed"        # Update with the path to your processed data directory
    output_directory = "data\\augmented"       # Update with the desired path for augmented data
    augment_factor = 2  # Adjust as needed

    augment_images(input_directory, output_directory, augment_factor)
