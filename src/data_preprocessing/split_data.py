# src/data_preprocessing/split_data.py

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # If not installed, run: pip install tqdm

def split_dataset(input_dir, output_dir, test_size=0.2, random_seed=42):
    """
    Split the dataset into training and testing sets and save them to the output directory.

    Parameters:
    - input_dir (str): Path to the directory containing class folders in the dataset.
    - output_dir (str): Path to the directory where split datasets will be saved.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_seed (int): Seed for random number generation for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of class folders
    classes = os.listdir(input_dir)

    # Split and save each class
    for class_name in tqdm(classes, desc="Splitting Classes"):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Create train and test subdirectories
        os.makedirs(os.path.join(output_class_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_class_dir, "test"), exist_ok=True)

        # List all image files in the class directory
        image_files = [f for f in os.listdir(input_class_dir) if f.endswith(('.JPG', '.JPEG', '.PNG'))]

        # Split the image files
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_seed)

        # Copy training set
        for filename in train_files:
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_class_dir, 'train', filename)
            shutil.copy(input_path, output_path)

        # Copy test set
        for filename in test_files:
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_class_dir, 'test', filename)
            shutil.copy(input_path, output_path)

if __name__ == "__main__":
    # Example usage
    input_directory = "data\\augmented"    # Update with the path to your augmented data directory
    output_directory = "data\\splits"       # Update with the desired path for split augmented data

    split_dataset(input_directory, output_directory)
