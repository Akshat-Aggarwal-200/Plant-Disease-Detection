import os
import matplotlib.pyplot as plt
from PIL import Image
import random

def load_sample_images(data_dir, num_samples_per_class=5):
    """
    Load and display a random sample of images from each class in the dataset.

    Parameters:
    - data_dir (str): Path to the directory containing class folders in the dataset.
    - num_samples_per_class (int): Number of samples to display for each class.
    """
    # Get the list of class folders
    classes = os.listdir(data_dir)

    # Display a random sample of images for each class
    plt.figure(figsize=(15, len(classes) * 3))
    for i, class_name in enumerate(classes, 1):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.JPG', '.JPEG', '.PNG'))]
        sample_files = random.sample(image_files, min(num_samples_per_class, len(image_files)))
    
        for j, filename in enumerate(sample_files, 1):
            image_path = os.path.join(class_dir, filename)
            img = Image.open(image_path)
            plt.subplot(len(classes), num_samples_per_class, (i - 1) * num_samples_per_class + j)
            plt.imshow(img)
            plt.title(f"Class: {class_name}\nSample {j}")
            plt.axis("off")
    plt.show()

def explore_data_distribution(data_dir):
    """
    Explore and visualize the class distribution of the dataset.

    Parameters:
    - data_dir (str): Path to the directory containing class folders in the dataset.
    """
    # Get the list of class folders
    classes = os.listdir(data_dir)

    # Count the number of images in each class
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

    # Visualize the class distribution
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_directory = "data\\raw"  # Update with the path to your raw data directory
    load_sample_images(data_directory)
    explore_data_distribution(data_directory)
