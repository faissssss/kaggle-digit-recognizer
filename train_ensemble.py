import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from dataloader import load_and_preprocess_data


# ── Model Architectures ─────────────────────────────────────────

def build_model_a():
    """2-block CNN: 32 → 64 filters (original architecture)"""
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])


def build_model_b():
    """3-block CNN: 32 → 64 → 128 filters (deeper)"""
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])


def build_model_d():
    """Wider 2-block CNN: 64 → 128 filters"""
    return Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])


# ── Training Configuration ──────────────────────────────────────

MODELS_TO_TRAIN = [
    {"name": "model_a_seed42",  "builder": build_model_a, "seed": 42},
    {"name": "model_b_seed42",  "builder": build_model_b, "seed": 42},
    {"name": "model_a_seed123", "builder": build_model_a, "seed": 123},
    {"name": "model_d_seed42",  "builder": build_model_d, "seed": 42},
    {"name": "model_b_seed99",  "builder": build_model_b, "seed": 99},
]


def get_augmenter():
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
    )


def train_single_model(config, X_train, y_train, X_val, y_val):
    """Trains a single model with the given config and returns the saved path."""
    name = config["name"]
    seed = config["seed"]
    save_path = f"ensemble_{name}.keras"

    # Skip if already trained
    if os.path.exists(save_path):
        print(f"\n{'='*60}")
        print(f"  SKIPPING {name} — already trained ({save_path})")
        print(f"{'='*60}")
        return save_path

    print(f"\n{'='*60}")
    print(f"  TRAINING: {name} (seed={seed})")
    print(f"{'='*60}")

    # Set seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = config["builder"]()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    datagen = get_augmenter()
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  {name} — Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    model.save(save_path)
    print(f"  Saved to {save_path}")

    return save_path


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    X_data, y_data, _ = load_and_preprocess_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.10, random_state=42
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    saved_models = []
    for config in MODELS_TO_TRAIN:
        path = train_single_model(config, X_train, y_train, X_val, y_val)
        saved_models.append(path)

    print(f"\n{'='*60}")
    print(f"  ALL MODELS TRAINED!")
    print(f"  Saved: {saved_models}")
    print(f"{'='*60}")
