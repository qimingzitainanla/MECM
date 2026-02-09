# MeCM-EDNet​
While emotions have proven to play a crucial role in decision-making process, the influence of emotions is often overlooked in current decision prediction models. To compensate for the lack of emotional integration, this paper proposes a single-trial decision prediction model, MeCM-EDNet, an emotional decision network (EDNet) that incorporates meiosis data augmentation (Me), contrastive learning (C), and multi-task learning (M). It consists of four blocks: data augmentation block, time learning block, graph learning block, and multi-task learning block. Among them, the multi-task learning block integrates decision prediction, emotion recognition, and supervised contrast learning tasks, effectively achieving the emotional integration of decision model. Moreover, we embed the neural mechanisms of emotion-driven decision-making into our graph learning module to overcome the limitations of decision models lacking neuroscientific knowledge. To validate our model, the paper designs an experiment to investigate how emotions influence spatial decision-making and constructs a dataset. Comparative experiments in our dataset and Emoback dataset demonstrate that MeCM-EDNet achieves state-of-the-art performance. The ablation studies confirm the effectiveness of each block of our model, particularly highlighting the important role of the integration of emotional information and neuroscience knowledge. MeCM-EDNet underscores the importance of considering emotional influence in decision research, addressing the current lack of emotional integration in most decision prediction models.

## 1. Environment Dependencies
This project is developed based on Python. The detailed environment dependencies and their versions are listed below, and you can also install them with the provided configuration file.
### 1.1 Core Dependencies

```bash
scipy==1.10.1
pandas==2.0.3
numpy==1.24.4
torch==2.1.0+cu118
einops==0.7.0
scikit-learn==1.3.2
tqdm==4.66.2
torcheeg==1.1.1
mne==1.6.1
torchaudio==2.1.0+cu118
torchvision==0.16.0+cu118
```

### 1.2 One-Click Installation
We provide two ways to install the dependencies (choose one according to your environment):

#### Option 1: Using pip (Python)

```bash
# Install dependencies via requirements.txt
pip install -r requirements_MECM.txt
```

#### Option 2: Using Conda (Recommended for GPU environment)
```bash
# Create a new conda environment
conda create -n <Environment name, e.g., model-reproduce> python=<Python version>
conda activate <Environment name>

# Install dependencies via environment.yml (if provided)
conda env update --file environment.yml

# Or install via pip if environment.yml is not used
pip install -r requirements_MECM.txt
```
## 2. Data Preparation
### 2.1 Data Format Description
The data format is consistent for both original and simulated data, as follows:

- Data Type: EEG signals / label data of emotion and decision-making behavior
- EEG Dimension: 62 channels × 875 time points (Sampling rate: 250 Hz)
- **Label Sets**: Two independent label sets are constructed to support multi-task learning of MeCM-EDNet:
  - `emo_label`: Label set divided by emotion categories
    - Label definition: 0 = neutral, 1 = happy, 2 = fear, 3 = sad
  - `dec_label`: Label set divided by decision correctness
    - Label definition: 0 = correct decision, 1 = incorrect decision
   
### 2.2 Train/Test Data
```bash
data/processed/
├── train/          
│   ├── eeg_data.npy  # EEG features (shape: [n_train, 1, 62, 875])
│   ├── emo_label.npy # Emotion labels (shape: [n_train, ])
│   └── dec_label.npy # Decision correctness labels (shape: [n_train, ])
└── test/           
    ├── eeg_data.npy  # Same structure as train set
    ├── emo_label.npy
    └── dec_label.npy
```

### 2.3 Simulated Data Generation Rules
```bash
import os
import numpy as np
from tqdm import tqdm

# Configuration parameters (adjustable according to requirements)
CONFIG = {
    # Data dimensions (matches your requirement: [n_samples, 1, 62, 875])
    "eeg_shape": (1, 62, 875),
    # Number of samples for train/test sets (8:2 ratio, customizable)
    "n_train_samples": 800,
    "n_test_samples": 200,
    # Emotion label categories (0=neutral, 1=happy, 2=fear, 3=sad)
    "emo_label_classes": [0, 1, 2, 3],
    # Decision label categories (0=correct, 1=incorrect)
    "dec_label_classes": [0, 1],
    # Output root directory
    "output_root": "data/processed/",
    # Random seed (ensures data reproducibility)
    "random_seed": 42
}

def create_directory_structure(output_root):
    """Create the specified directory structure"""
    train_dir = os.path.join(output_root, "train")
    test_dir = os.path.join(output_root, "test")
    
    # Create directories if they do not exist
    for dir_path in [train_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    return train_dir, test_dir

def generate_simulated_eeg_data(n_samples, eeg_shape, random_state):
    """Generate simulated EEG data (matching the value range of real EEG signals)
    Args:
        n_samples: Number of samples to generate
        eeg_shape: Shape of single EEG sample (1, 62, 875)
        random_state: Numpy random state for reproducibility
    Returns:
        Simulated EEG data with shape (n_samples, 1, 62, 875)
    """
    # Simulated EEG signal range: typically between -100 and 100 microvolts
    eeg_data = random_state.normal(
        loc=0,  # Mean value
        scale=50,  # Standard deviation
        size=(n_samples,) + eeg_shape  # Total shape: [n_samples, 1, 62, 875]
    )
    # Clip values to match real EEG signal range
    eeg_data = np.clip(eeg_data, -100, 100)
    return eeg_data.astype(np.float32)  # Use float32 to save storage space

def generate_labels(n_samples, label_classes, random_state):
    """Generate simulated labels with uniform distribution
    Args:
        n_samples: Number of labels to generate
        label_classes: List of possible label values
        random_state: Numpy random state for reproducibility
    Returns:
        Array of generated labels with shape (n_samples,)
    """
    return random_state.choice(
        label_classes,
        size=n_samples,
        replace=True  # Allow duplicate labels
    ).astype(np.int64)

def save_data(data, label_emo, label_dec, save_dir, prefix):
    """Save data to the specified directory
    Args:
        data: EEG data array
        label_emo: Emotion label array
        label_dec: Decision label array
        save_dir: Target directory for saving
        prefix: Prefix for data files (train/test)
    """
    # Save EEG data
    eeg_path = os.path.join(save_dir, f"{prefix}_data.npy")
    np.save(eeg_path, data)
    
    # Save emotion labels
    emo_label_path = os.path.join(save_dir, "emo_label.npy")
    np.save(emo_label_path, label_emo)
    
    # Save decision labels
    dec_label_path = os.path.join(save_dir, "dec_label.npy")
    np.save(dec_label_path, label_dec)
    
    print(f"✅ {prefix.capitalize()} data saved to {save_dir}")
    print(f"   - EEG shape: {data.shape}")
    print(f"   - Emotion labels shape: {label_emo.shape}")
    print(f"   - Decision labels shape: {label_dec.shape}\n")

def main():
    # Set random seed to ensure reproducibility
    random_state = np.random.RandomState(CONFIG["random_seed"])
    
    # 1. Create directory structure
    print("🔧 Creating directory structure...")
    train_dir, test_dir = create_directory_structure(CONFIG["output_root"])
    
    # 2. Generate training set data
    print("📊 Generating training data...")
    train_eeg = generate_simulated_eeg_data(
        CONFIG["n_train_samples"],
        CONFIG["eeg_shape"],
        random_state
    )
    train_emo_labels = generate_labels(
        CONFIG["n_train_samples"],
        CONFIG["emo_label_classes"],
        random_state
    )
    train_dec_labels = generate_labels(
        CONFIG["n_train_samples"],
        CONFIG["dec_label_classes"],
        random_state
    )
    save_data(train_eeg, train_emo_labels, train_dec_labels, train_dir, "train")
    
    # 3. Generate test set data
    print("📊 Generating test data...")
    test_eeg = generate_simulated_eeg_data(
        CONFIG["n_test_samples"],
        CONFIG["eeg_shape"],
        random_state
    )
    test_emo_labels = generate_labels(
        CONFIG["n_test_samples"],
        CONFIG["emo_label_classes"],
        random_state
    )
    test_dec_labels = generate_labels(
        CONFIG["n_test_samples"],
        CONFIG["dec_label_classes"],
        random_state
    )
    save_data(test_eeg, test_emo_labels, test_dec_labels, test_dir, "test")
    
    print("🎉 All simulated data generated successfully!")

if __name__ == "__main__":
    main()
```
## 3. Model Usage
After completing the data preparation (generating simulated data or preprocessing raw data), you can directly run the Jupyter Notebook file to train and evaluate the MeCM-EDNet model.

### 3.1 Prerequisites
Ensure the following conditions are met before running the notebook:
- The environment dependencies are installed correctly (see Section 1: Environment Dependencies)
- The preprocessed data is saved in the `data/processed/` directory (see Section 2: Data Preparation)
- Jupyter Notebook is installed (if not, run `pip install notebook`)

### 3.2 Run the Notebook
1. Launch Jupyter Notebook in the repository root directory:
   ```bash
   jupyter notebook
   ```
2.In the Jupyter interface, navigate to the root directory of the repository and open the MECM_EDNet.ipynb file.

3.Run all cells sequentially (or run cell by cell to debug step by step):

- The notebook includes complete workflows: data loading, model initialization, training, evaluation, and result output.
- All hyperparameters (learning rate, batch size, epochs, etc.) are predefined in the notebook (consistent with the paper's experimental settings).
