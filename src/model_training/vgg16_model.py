# src/models/train_vgg16.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from utils.train_utils import load_data, train_model, evaluate_model

if __name__ == "__main__":
    input_directory = ".data\\split"
    output_directory = "model_saved"
    no_of_clases = len(os.listdir(os.path.join(input_directory, 'train')))

    # Load and train VGG16
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(256,256, 3))
    vgg16_model = Sequential([
        vgg16,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(no_of_clases, activation='softmax')
    ])
    vgg16_train_generator = load_data(input_directory, input_size=(256,256), subset='training')
    vgg16_validation_generator = load_data(input_directory, input_size=(256,256), subset='validation')
    train_model(vgg16_model, vgg16_train_generator, vgg16_validation_generator, os.path.join(output_directory, 'vgg16_model.h5'))
