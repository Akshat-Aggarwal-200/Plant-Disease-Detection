# src/data_preprocessing/split_data.py

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # If not installed, run: pip install tqdm

def split_data(input_dir, output_dir, test_size=0.2):
    """
    Split the augmented data into training and testing sets and organize the data accordingly.

    Parameters:
    - input_dir (str): Path to the directory containing augmented data.
    - output_dir (str): Path to the directory where the split data will be saved.
    - test_size (float): Fraction of the dataset to include in the test split (e.g., 0.2 for 20%).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of class folders
    classes = os.listdir(input_dir)

    for class_name in tqdm(classes, desc="Splitting Data"):
        input_class_dir = os.path.join(input_dir, class_name)
        output_train_class_dir = os.path.join(output_dir, "train", class_name)
        output_test_class_dir = os.path.join(output_dir, "test", class_name)

        os.makedirs(output_train_class_dir, exist_ok=True)
        os.makedirs(output_test_class_dir, exist_ok=True)

        # List all image files in the class directory
        image_files = [f for f in os.listdir(input_class_dir) if f.endswith(('.jpg', '.jpeg', '.png','.JPG','.JPEG','.PNG'))]

        # Split data into training and testing sets
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

        # Move training set to the output directory
        for filename in train_files:
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_train_class_dir, filename)
            shutil.copy(input_path, output_path)

        # Move testing set to the output directory
        for filename in test_files:
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_test_class_dir, filename)
            shutil.copy(input_path, output_path)

if __name__ == "__main__":
    # Example usage
    input_directory = "data\\augmented"  # Update with the path to your augmented data directory
    output_directory = "data\\split"  # Update with the desired path for split data

    split_data(input_directory, output_directory, test_size=0.2)
