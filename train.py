import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataloader import load_and_preprocess_data


def build_model():
    """
    Builds a deep CNN for digit classification.
    Architecture:
      - 2x Conv2D(32) + BN + MaxPool + Dropout(0.25)
      - 2x Conv2D(64) + BN + MaxPool + Dropout(0.25)
      - Flatten + Dense(256) + BN + Dropout(0.5)
      - Dense(10, softmax)
    """
    model = Sequential([
        # --- Block 1: 32 filters ---
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Block 2: 64 filters ---
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Classifier Head ---
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history):
    """Plots accuracy and loss curves and saves to training_history.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history plot saved to training_history.png")


if __name__ == "__main__":
    # --- Step 2: Load & preprocess ---
    X_data, y_data, X_test = load_and_preprocess_data()

    # Split into train and validation sets (85/15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.15, random_state=42
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # --- Data Augmentation ---
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(X_train)

    # --- Step 3: Build model ---
    model = build_model()
    model.summary()

    # --- Step 4: Train ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluate ---
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print(f"Final Validation Loss:     {val_loss:.4f}")

    # Save model weights
    model.save('digit_model.keras')
    print("Model saved to digit_model.keras")

    # Plot training curves
    plot_training_history(history)
