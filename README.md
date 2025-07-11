
# Enhancing AASIST for Robust Speech Deepfake Detection

This repository contains the official PyTorch implementation for the paper **"Enhancing AASIST for Robust Speech Deepfake Detection"**.

In this work, we introduce a series of targeted architectural refinements to the AASIST framework to improve its robustness against sophisticated spoofing attacks.

Our enhanced model demonstrates promising performance on the **ASVspoof 5 dataset**, achieving an **Equal Error Rate (EER) of 7.6%** without external data in training dataset.

## Key Innovations

Our model introduces the following enhancements to the AASIST baseline architecture:

1.  **Robust Feature Extraction**:
    *   We use a **frozen Wav2Vec 2.0 encoder** as a fixed feature extractor. This preserves its powerful, generalized speech representations learned from vast unlabeled data and prevents feature degradation (catastrophic forgetting) during fine-tuning on the smaller, task-specific dataset.

2.  **Adaptive Attention Mechanisms**:
    *   The custom graph attention module is replaced with the standardized and highly optimized `torch.nn.MultiheadAttention`, enabling the use of hardware-accelerated backends like Flash Attention for significant speedups.
    *   We implement a **Heterogeneous Attention** module with node-type-specific linear projections for `Query` and `Key` tensors. This allows the model to learn more nuanced interaction protocols between temporal (T) and spectral (S) nodes.

3.  **Learned Fusion Strategy**:
    *   The heuristic-based `torch.max` operation for fusing parallel processing streams is replaced with a **learnable, attention-based fusion mechanism**. This mitigates information bottlenecks by allowing the model to adaptively weigh and integrate features from both streams in a context-aware manner.

4.  **Advanced Training Procedure**:
    *   We employ a comprehensive augmentation pipeline that simulates real-world distortions, including **codec compression** (MP3, low-bitrate) and **acoustic effects** (noise, reverberation).
    *   A **dynamic learning strategy** transitions from Cross-Entropy to Focal Loss based on validation performance, progressively focusing the model on more challenging samples and preventing late-stage overfitting.

## Installation

To get started, clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git # TODO: Replace with your repo URL
cd your-repo-name

# 2. Create and activate a virtual environment (recommended)
conda create -n aasist python=3.13 --no-default-packages -y
conda activate aasist
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install dependencies
pip install uv
uv pip install -r requirements.txt
```

This project uses `accelerate` for distributed training. If you haven't used it before, run the configuration wizard:
```bash
accelerate config
```
Follow the interactive prompts to set up your training environment (e.g., single GPU, multi-GPU).

## Training

The model training is launched using the `train_accelerate.py` script via the `accelerate` launcher.

```bash
accelerate launch experiments/train_accelerate.py
```

All experiment configurations, including hyperparameters, data paths, and augmentation settings, are defined in configuration files (e.g., within a `configs/` directory). Before running, please ensure the paths to your dataset are correctly specified in the relevant config file.

## Results

Our model achieved the following performance on the ASVspoof 5 evaluation set:

| Metric      | Value |
| :---------- | :---: |
| **EER (%)** | **7.6** |


## Citation

If you use this code or the ideas from our paper in your research, please cite our work:

```bibtex
@misc{viakhirev2024enhancing,
      title={Enhancing AASIST for Robust Speech Deepfake Detection}, 
      author={Ivan Viakhirev and Daniil Sirota and Aleksandr Smirnov and Kirill Borodin},
      year={2025},
      eprint={24XX.XXXXX},  -- TODO: Replace with the actual arXiv ID
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.