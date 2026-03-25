import numpy as np
import pandas as pd
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataloader import load_and_preprocess_data


def tta_predict_single(model, X_test, n_augments=15, batch_size=128):
    """TTA for a single model — averages base + augmented predictions."""
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
    )

    # Base predictions
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)

    # Augmented predictions
    for i in range(n_augments):
        augmented = np.array([datagen.random_transform(img) for img in X_test])
        preds += model.predict(augmented, batch_size=batch_size, verbose=0)

    return preds / (n_augments + 1)


def ensemble_predict(model_paths, X_test, use_tta=True, n_augments=15):
    """Loads all models, runs TTA on each, averages across all models."""
    all_predictions = []

    for i, path in enumerate(model_paths):
        print(f"\nModel {i+1}/{len(model_paths)}: {path}")
        model = load_model(path)

        if use_tta:
            print(f"  Running TTA ({n_augments} augmentation passes)...")
            preds = tta_predict_single(model, X_test, n_augments=n_augments)
        else:
            preds = model.predict(X_test, batch_size=128, verbose=0)

        all_predictions.append(preds)
        print(f"  Done.")

    # Average across all models
    avg_predictions = np.mean(all_predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)


if __name__ == "__main__":
    # Find all ensemble model files
    model_paths = sorted(glob.glob("ensemble_*.keras"))

    if not model_paths:
        print("No ensemble models found! Run train_ensemble.py first.")
        exit(1)

    print(f"Found {len(model_paths)} ensemble models: {model_paths}")

    print("\nLoading test data...")
    _, _, X_test = load_and_preprocess_data()

    print("\nGenerating ensemble + TTA predictions...")
    predicted_labels = ensemble_predict(model_paths, X_test, use_tta=True, n_augments=15)

    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })

    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission saved to submission.csv ({len(submission)} predictions)")
    print(f"\nSample predictions:\n{submission.head(10)}")
