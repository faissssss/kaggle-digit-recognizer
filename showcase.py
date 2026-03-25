import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_showcase(test_csv='test.csv', sub_csv='submission.csv', output_img='showcase.png'):
    """
    Creates a visual grid of test images and the model's predicted labels.
    Perfect for GitHub READMEs and presentations.
    """
    if not os.path.exists(test_csv) or not os.path.exists(sub_csv):
        print(f"Error: {test_csv} or {sub_csv} not found.")
        return

    print("Loading data for showcase...")
    test_df = pd.read_csv(test_csv)
    sub_df = pd.read_csv(sub_csv)

    # Pick 20 random samples to showcase
    indices = np.random.choice(len(test_df), 20, replace=False)
    
    plt.figure(figsize=(15, 6))
    plt.suptitle("Ensemble CNN Model Predictions (MNIST Digit Recognizer)", fontsize=18, fontweight='bold', y=1.05)
    
    for i, idx in enumerate(indices):
        plt.subplot(4, 5, i + 1)
        
        # Reshape the 784 pixels back to 28x28
        img = test_df.iloc[idx].values.reshape(28, 28)
        label = sub_df.iloc[idx]['Label']
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {label}", fontsize=12, color='blue', fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"Showcase image saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    create_showcase()
