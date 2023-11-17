# src/models/train_custom.py

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from utils.train_utils import load_data, train_model, evaluate_model

if __name__ == "__main__":
    input_directory = ".data\\split"
    output_directory = "model_saved"
    no_of_clases = len(os.listdir(os.path.join(input_directory, 'train')))

    # Define and train custom TensorFlow model
    custom_model = Sequential([
        # Define your custom model architecture here
        # Example:
        Conv2D(16,(3,3),1,activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32,(3,3),1,activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32,(3,3),1,activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(16,(3,3),1,activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(no_of_clases, activation='softmax')
    ])
    custom_train_generator = load_data(input_directory, input_size=(256, 256), subset='training')
    custom_validation_generator = load_data(input_directory, input_size=(256, 256), subset='validation')
    train_model(custom_model, custom_train_generator, custom_validation_generator, os.path.join(output_directory, 'custom_model.h5'))
