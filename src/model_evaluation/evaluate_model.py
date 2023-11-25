import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def evaluate_trained_models(models_directory, test_data_directory):
    """
    Evaluate trained models on the test set and print the evaluation metrics.

    Parameters:
    - models_directory (str): Path to the directory containing trained model files.
    - test_data_directory (str): Path to the directory containing the test data.
    """
    # Get the list of trained model files
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]

    for model_file in tqdm(model_files, desc="Evaluating Models"):
        model_path = os.path.join(models_directory, model_file)
        model_name = os.path.splitext(model_file)[0]

        print(f"\nEvaluating Model: {model_name}")

        # Load the model
        model = load_model(model_path)

        # Create a data generator for the test set
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_directory,
            target_size=(256,256),
            batch_size=32,
            class_mode='sparse',
            shuffle=False
        )

        # Evaluate the model on the test set
        evaluate_model(model, test_generator)

def evaluate_model(model, generator):
    """
    Evaluate the trained model on the given generator and print the evaluation metrics.

    Parameters:
    - model: Trained Keras model.
    - generator: Data generator for the test set.
    """
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

if __name__ == "__main__":
    # Example usage
    trained_models_directory = "src\\model_saved"  # Update with the path to your trained models directory
    test_data_directory = "data\\split\\test"  # Update with the path to your test data directory

    evaluate_trained_models(trained_models_directory, test_data_directory)
