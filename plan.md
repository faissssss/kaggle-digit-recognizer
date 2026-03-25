# Digit Recognizer (MNIST) Implementation Plan

## 1. Goal and Problem Understanding
The objective is to achieve maximum classification accuracy when identifying handwritten digits (0-9) from 28x28 pixel grayscale image inputs. 

**Data Available:**
- `train.csv` (42,000 samples). Includes `label` (0-9) and `pixel0` through `pixel783`.
- `test.csv` (28,000 samples). Includes `pixel0` through `pixel783`.

**Metric:** Categorization Accuracy.

## 2. Infrastructure Setup
### Environment
- Language: Python 3
- Libraries: `pandas`, `numpy`, `tensorflow/keras`, `scikit-learn`, `matplotlib`, `seaborn`
- CLI: `kaggle` for data retrieval and submission

### Project Structure (Proposed)
```text
digit-recognizer/
│
├── requirements.txt      # Project dependencies
├── plan.md               # Infrastructure and workflow plan
├── download_data.py      # Script to automate downloading from Kaggle
├── train.py              # Main training script
├── predict.py            # Generates submission.csv on test.csv
└── dataloader.py         # Handles CSV loading and preprocessing
```

## 3. Workflow Steps

### Step 1: Data Acquisition
- Run Kaagle CLI: `kaggle competitions download -c digit-recognizer`
- Unzip `train.csv` and `test.csv` into a local `data/` directory.

### Step 2: Data Loading & Preprocessing
- **Load files**: Use `pandas.read_csv` to load data efficiently into memory.
- **Separate Features & Labels**: In `train.csv`, `y_train` comes from the `label` column. `X_train` comes from the remaining 784 pixel columns.
- **Normalization**: Pixel values range from `[0, 255]`. Divide all pixel arrays by `255.0` to reach a `[0.0, 1.0]` scale. This speeds up convergence.
- **Reshaping**: Since images inherently have spatial structure, reshape the 1D arrays of 784 elements to 3D arrays of shape `(28, 28, 1)` for Convolutional Neural Networks (CNNs).
- **Label Encoding**: Apply one-hot encoding using `keras.utils.to_categorical` to `y_train` for 10 classes.

### Step 3: Model Architecture Configuration (CNN)
We will build a deep CNN capable of automated feature extraction on spatial data.
- **Conv2D Layers**: Several layers with ReLU activation to learn spatial filters (edges, curves).
- **MaxPooling2D**: To downsample the feature dimension, retaining only max activations.
- **Dropout Layers**: Regularization technique (e.g., dropping 25% or 50% nodes) to prevent the model from overfitting on 42,000 images.
- **Flatten Layer**: Transforms 2D spatial maps back to 1D vectors for classification.
- **Dense Output Layer**: A soft-max activation layer with 10 units representing the probabilities of digits 0 through 9.

### Step 4: Training Setup
- **Optimizer**: `Adam` (adaptive learning rate)
- **Loss Function**: `categorical_crossentropy` (standard for multi-class classification)
- **Metric**: `accuracy`
- **Validation**: Extract a 10%-20% holdout validation set out of `train.csv` using `train_test_split` to monitor overfitting iteratively.
- **Callbacks**: Implementing `EarlyStopping` or `ReduceLROnPlateau` for automated tuning.

### Step 5: Evaluation & Prediction
- Monitor the Model Accuracy graph generated periodically by `matplotlib`.
- After finalizing weights on the validation set, load `test.csv`, apply identical preprocessing (Normalize to `0.0/1.0`, Reshape to `(28,28,1)`).
- Model outputs raw probabilities. We use `numpy.argmax` to extract the most probable integer class `(0-9)`.

### Step 6: Submission Export
- Transform `argmax` predictions into a `pandas` DataFrame with `ImageId` (1 to 28000) and `Label`.
- Export to `submission.csv` preserving headers and skipping indexes.
- Submit via Kaggle CLI: `kaggle competitions submit -c digit-recognizer -f submission.csv -m "Initial CNN Submission"`

## 4. Verification Checkpoints
1. Do loaded feature matrices match shape `(42000, 784)` and labels match `(42000,)`?
2. Post-reshape, does an arbitrary sample plot with `matplotlib.pyplot.imshow()` resemble a real, recognizable digit number?
3. Does validation loss smoothly decrease over epochs? Any signs of explosive overfitting?
4. Does the generated `submission.csv` contain exactly 28,000 predictions?
