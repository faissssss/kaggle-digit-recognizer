import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(train_path='train.csv', test_path='test.csv'):
    """
    Loads digit recognizer CSV data, normalizes, reshapes, and one-hot encodes labels.
    Returns: X_train, y_train, X_test
    """
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate labels from features in training data
    y_train_raw = train_df['label'].values
    X_train_raw = train_df.drop('label', axis=1).values
    
    # Test data contains no labels
    X_test_raw = test_df.values
    
    print("Normalizing pixel values...")
    # Normalize values from [0, 255] to [0.0, 1.0]
    X_train_norm = X_train_raw / 255.0
    X_test_norm = X_test_raw / 255.0
    
    print("Reshaping arrays to (28, 28, 1)...")
    # Reshape features for 2D Convolutions
    X_train = X_train_norm.reshape(-1, 28, 28, 1)
    X_test = X_test_norm.reshape(-1, 28, 28, 1)
    
    print("One-hot encoding labels...")
    # Convert labels 0-9 to one-hot encoded vectors
    y_train = to_categorical(y_train_raw, num_classes=10)
    
    print(f"Data ready! X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train, X_test

if __name__ == "__main__":
    # Quick sanity check
    X_train, y_train, X_test = load_and_preprocess_data()
