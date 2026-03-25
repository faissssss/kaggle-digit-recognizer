import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataloader import load_and_preprocess_data


def tta_predict(model, X_test, n_augments=10, batch_size=128):
    """
    Test Time Augmentation: generates multiple augmented versions of each
    test image, averages the prediction probabilities, then picks the
    most confident class. Typically adds +0.1-0.2% accuracy.
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    # Start with un-augmented base predictions
    print("  Base predictions (no augmentation)...")
    predictions_sum = model.predict(X_test, batch_size=batch_size, verbose=0)

    # Add augmented predictions
    for i in range(n_augments):
        print(f"  TTA pass {i + 1}/{n_augments}...")
        augmented = np.array([datagen.random_transform(img) for img in X_test])
        predictions_sum += model.predict(augmented, batch_size=batch_size, verbose=0)

    # Average across all passes (1 base + n_augments)
    avg_predictions = predictions_sum / (n_augments + 1)
    return np.argmax(avg_predictions, axis=1)


def generate_submission(model_path='digit_model.keras', output_path='submission.csv', use_tta=True):
    """
    Loads the trained model and test data, generates predictions
    (with optional TTA), and exports a Kaggle-format submission CSV.
    """
    print("Loading trained model...")
    model = load_model(model_path)

    print("Loading and preprocessing test data...")
    _, _, X_test = load_and_preprocess_data()

    if use_tta:
        print(f"Generating TTA predictions (10 augmentation passes)...")
        predicted_labels = tta_predict(model, X_test, n_augments=10)
    else:
        print("Generating standard predictions...")
        predictions = model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)

    print("Creating submission file...")
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} predictions)")
    print(f"\nSample predictions:\n{submission.head(10)}")

    return submission


if __name__ == "__main__":
    generate_submission(use_tta=True)
