import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(input_dir, input_size=(224, 224), batch_size=32, subset='training'):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    generator = datagen.flow_from_directory(
        input_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset
    )

    return generator

def train_model(model, train_generator, validation_generator, output_model_path):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=5, validation_data=validation_generator)

    # Save the trained model
    model.save(output_model_path)

    # Evaluate the model
    evaluate_model(model, validation_generator)

def evaluate_model(model, generator):
    # Predictions
    predictions = model.predict(generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # True labels
    true_labels = generator.classes

    # Class labels
    class_labels = list(generator.class_indices.keys())

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=class_labels))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_classes))
