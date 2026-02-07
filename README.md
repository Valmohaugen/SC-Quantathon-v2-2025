# SC-Quantathon v2 (2025)

Welcome to the SC-Quantathon v2 repository! This project is part of the 2025 SC Quantum Hackathon, focusing on applying quantum machine learning (QML) techniques to real-world scientific challenges. The repository contains code, data, and results for quantum and classical machine learning models, with a focus on tornado prediction using quantum-enhanced algorithms.

## Project Overview

This repository explores the use of quantum machine learning for binary and multi-class classification tasks related to tornado prediction. The project leverages multiple quantum computing frameworks (PennyLane, Qiskit, Cirq) and compares quantum models to classical baselines.

## Directory Structure

```
.
├── README.md
├── tornado_dataset.py
├── tornadont.ipynb
├── QNN/
│   ├── QNN_binary.py
│   └── PennyLane/
│       ├── classification_report.txt
│       ├── confusion_matrix.csv
│       └── qnn_model.pth
├── QSVM/
│   ├── plot_qsvm_comparison.py
│   ├── QSVM_binary.py
│   ├── QSVM_multi.py
│   ├── quantum_features_ml.py
│   ├── Cirq/
│   │   ├── best_params.txt
│   │   ├── QSVM_binary_classification_report.txt
│   │   ├── QSVM_binary_confusion_matrix.csv
│   │   ├── QSVM_binary_X_train.npy
│   │   ├── QSVM_multi_best_params.txt
│   │   ├── QSVM_multi_classification_report.txt
│   │   └── QSVM_multi_confusion_matrix.csv
│   ├── PennyLane/
│   │   └── ... (same as above)
│   └── Qiskit/
│       └── ... (same as above)
```

- **QNN/**: Quantum Neural Network models and results.
- **QSVM/**: Quantum Support Vector Machine models, scripts, and results.
- **tornado_dataset.py**: Script for loading and preprocessing the tornado dataset.
- **tornadont.ipynb**: Main notebook for data exploration, model training, and evaluation.

## Dataset

The dataset is related to tornado prediction, likely containing meteorological features and tornado occurrence labels. Data loading and preprocessing are handled in `tornado_dataset.py` and explored in `tornadont.ipynb`.

## Quantum Models

### Quantum Neural Network (QNN)

- **QNN_binary.py**: Implements a binary classification QNN using PennyLane.
- **QNN/PennyLane/**: Stores trained model weights (`qnn_model.pth`), classification reports, and confusion matrices.

### Quantum Support Vector Machine (QSVM)

- **QSVM_binary.py**: Binary classification using QSVM.
- **QSVM_multi.py**: Multi-class classification using QSVM.
- **quantum_features_ml.py**: Feature engineering and classical ML baselines for comparison.
- **QSVM/[Cirq|PennyLane|Qiskit]/**: Results and best parameters for each quantum framework.

#### Frameworks

- **PennyLane**: Hybrid quantum-classical ML with differentiable programming.
- **Qiskit**: IBM's quantum SDK for circuit-based QSVMs.
- **Cirq**: Google's quantum SDK for circuit-based QSVMs.

## Classical Baselines

Classical machine learning models are implemented in `quantum_features_ml.py` for benchmarking quantum models.

## Results & Reports

Each quantum framework directory contains:
- `best_params.txt`: Best hyperparameters found.
- `classification_report.txt`: Precision, recall, F1-score, and support.
- `confusion_matrix.csv`: Confusion matrix for model predictions.
- `X_train.npy`: Training data used for the quantum models.

## How to Run

1. **Install dependencies** (see below).
2. **Prepare the dataset** using `tornado_dataset.py`.
3. **Train and evaluate models**:
	 - Run `QNN/QNN_binary.py` for QNN experiments.
	 - Run `QSVM/QSVM_binary.py` or `QSVM/QSVM_multi.py` for QSVM experiments.
	 - Use `tornadont.ipynb` for interactive exploration and visualization.
4. **Compare results** using the reports and plots in each framework directory.

## Dependencies

- Python 3.8+
- PennyLane
- Qiskit
- Cirq
- NumPy, pandas, scikit-learn, matplotlib

Install dependencies with:

```bash
pip install pennylane qiskit cirq numpy pandas scikit-learn matplotlib
```

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/TariniHardikar/SCQuantum-SRNL-Challenge-2025.git)