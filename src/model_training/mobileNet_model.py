# src/models/train_mobilenet.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from utils.train_utils import load_data, train_model, evaluate_model

if __name__ == "__main__":
    input_directory = ".data\\split"
    output_directory = "model_saved"
    no_of_clases = len(os.listdir(os.path.join(input_directory, 'train')))

    # Load and train MobileNet
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    mobilenet_model = Sequential([
        mobilenet,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(len(os.listdir(os.path.join(input_directory, 'train'))), activation='softmax')
    ])
    mobilenet_train_generator = load_data(input_directory, input_size=(256, 256), subset='training')
    mobilenet_validation_generator = load_data(input_directory, input_size=(256, 256), subset='validation')
    train_model(mobilenet_model, mobilenet_train_generator, mobilenet_validation_generator, os.path.join(output_directory, 'mobilenet_model.h5'))
