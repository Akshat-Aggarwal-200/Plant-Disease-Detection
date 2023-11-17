import os
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import random

def load_sample_images(data_dir : str, num_samples = 5):
    """
    Load and display a random sample of images from the dataset.

    Parameters:
    - data_dir(str) : Path to dirctory containing the dataset.
    - num_samples (int) : Number of samples to display.
    """

    # List all image files in data dirctory
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg','.jpeg','.png'))]

    #Randomply select a subset of images
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    # Display sample images
    plt.figure(figsize=(15,5))
    for i,filename in enumerate(sample_files, 1):
        image_path = os.path.join(data_dir, filename)
        img = Image.open(image_path)
        plt.subplot(1, num_samples,i)
        plt.imshow(img)
        plt.title(f"sample {i}")
        plt.axis("off")
    plt.show

def explore_data_distribution(data_dir):
    """
    Explore and visualize the class distribution of the dataset.
    
    Parameters:
    - data_dir (str): Path to the directory containing the dataset.
    """
    # Assuming subdirectories in data_dir represent different classes
    classes = sorted(os.listdir(data_dir))

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
    # project_root = "D:\\Data Science\\Project\\Plant-Disease-Detection"
    data_directory = "data\\raw"  # Update with the path to your raw data directory
    load_sample_images(data_directory)
    explore_data_distribution(data_directory)