# MeCM-EDNet​
While emotions have proven to play a crucial role in decision-making process, the influence of emotions is often overlooked in current decision prediction models. To compensate for the lack of emotional integration, this paper proposes a single-trial decision prediction model, MeCM-EDNet, an emotional decision network (EDNet) that incorporates meiosis data augmentation (Me), contrastive learning (C), and multi-task learning (M). It consists of four blocks: data augmentation block, time learning block, graph learning block, and multi-task learning block. Among them, the multi-task learning block integrates decision prediction, emotion recognition, and supervised contrast learning tasks, effectively achieving the emotional integration of decision model. Moreover, we embed the neural mechanisms of emotion-driven decision-making into our graph learning module to overcome the limitations of decision models lacking neuroscientific knowledge. To validate our model, the paper designs an experiment to investigate how emotions influence spatial decision-making and constructs a dataset. Comparative experiments in our dataset and Emoback dataset demonstrate that MeCM-EDNet achieves state-of-the-art performance. The ablation studies confirm the effectiveness of each block of our model, particularly highlighting the important role of the integration of emotional information and neuroscience knowledge. MeCM-EDNet underscores the importance of considering emotional influence in decision research, addressing the current lack of emotional integration in most decision prediction models.

## 1. Environment Dependencies
This project is developed based on Python. The detailed environment dependencies and their versions are listed below, and you can also install them with the provided configuration file.
### 1.1 Core Dependencies
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
pip install -r requirements.txt
```
