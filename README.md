# SC-Quantathon-v2-2025: SRNL Challenge

Welcome to the SC-Quantathon-v2-2025 repository—a collaborative space for quantum machine learning and benchmarking. This project continues the Quantathon tradition, focusing on quantum ML, noise analysis, and practical applications for tornado prediction.

## Features

- **Quantum Machine Learning:** Train and evaluate quantum models (QNN, QSVM) for tornado prediction and classification. Compare quantum models to classical ML baselines.
- **Noise & Fidelity Characterization:** Analyze hardware-induced noise, decoherence, and gate errors. Visualize their impact on model performance using confusion matrices and classification reports.
- **Real-World Benchmarking:** Apply quantum ML output to practical benchmarks, demonstrating utility in scientific and meteorological scenarios.
- **Data Generation & Analysis:** Generate, process, and analyze tornado datasets. Scripts and notebooks provided for reproducibility.

## How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   ```

2. **Explore the project structure:**
   - `QNN/`: Quantum Neural Network models and results
   - `QSVM/`: Quantum Support Vector Machine models and results
   - `tornado_dataset.py`: Data generation and processing scripts
   - `tornadont.ipynb`: Main notebook for experiments and analysis

3. **Run notebooks and scripts:**
   - Use Jupyter Notebook or VS Code to run `.ipynb` files and Python scripts.
   - Each stage is documented for reproducibility.

4. **Install dependencies:**
   - Ensure you have Python, Jupyter, and required packages (e.g., numpy, scikit-learn, matplotlib, PennyLane, Qiskit, Cirq).
   - Add extra packages as needed for quantum ML and data analysis.

## Results

By following the stages, you will:

- Train quantum and classical classifiers for tornado prediction.
- Characterize and mitigate noise, improving model quality.
- Compare quantum and classical ML performance on meteorological data.
- Reproduce and extend quantum ML experiments with provided datasets and scripts.

## Project Stages

1. **Data Generation:** Generate tornado datasets using quantum and classical methods.
2. **Quantum ML Training:** Train QNN and QSVM models for classification and verification.
3. **Noise Analysis:** Analyze noise, fidelity, and quantum hardware effects using confusion matrices and reports.
4. **Benchmarking & Applications:** Apply results to real-world benchmarks and verify performance.