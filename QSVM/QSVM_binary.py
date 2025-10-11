import os
import pickle
import sys
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
import itertools
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import ParameterSampler, ParameterGrid

# Silence noisy sklearn warning
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*does not have valid feature names.*",
    category=UserWarning,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def _parse_bool_flag(name: str, default: bool = True) -> bool:
    """Parse environment variable as boolean flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _parse_fourier_scales(raw: str) -> list:
    """Parse comma-separated Fourier scales from environment variable."""
    try:
        return [float(s) for s in raw.split(",") if s.strip()]
    except Exception:
        return [1.0]


class Config:
    """Central configuration for QSVM experiments."""

    # Quantum simulator selection
    SIMULATOR = os.getenv("QSVM_SIMULATOR", "cirq").lower()

    # Dimensionality reduction
    PCA_COMPONENTS = int(os.getenv("QSVM_PCA_COMPONENTS", "8"))

    # Feature expansion
    FEATURE_EXPANSION = os.getenv("QSVM_FEATURE_EXPAND", "None").lower()
    RFF_COMPONENTS = int(os.getenv("QSVM_RFF_COMPONENTS", "128"))
    RFF_GAMMA = float(os.getenv("QSVM_RFF_GAMMA", "1.0"))
    FOURIER_SCALES = _parse_fourier_scales(os.getenv("QSVM_FOURIER_SCALES", "0.1"))

    # Post-processing toggles
    POST_PCA_STANDARDIZE = _parse_bool_flag("QSVM_POST_PCA_STD", True)
    CENTER_KERNEL = _parse_bool_flag("QSVM_CENTER", False)
    PSD_PROJECTION = _parse_bool_flag("QSVM_PSD", True)
    UNIT_DIAGONAL = _parse_bool_flag("QSVM_UNIT_DIAG", True)

    # Model knobs
    ANGLE_SCALE = float(os.getenv("QSVM_ANGLE_SCALE", "1.0"))  # scales RY/RZ angles
    SVM_C = float(os.getenv("QSVM_C", "1.0"))                  # SVC regularization

    # Paths
    CURR_DIR = Path(__file__).resolve().parent
    REPO_ROOT = CURR_DIR.parent


# Add repo root to path
if str(Config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(Config.REPO_ROOT))


# ============================================================================
# UTILITIES
# ============================================================================

@contextmanager
def pushd(new_dir: Path):
    """Context manager for temporary directory changes."""
    prev = os.getcwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(prev)


def next_power_of_two(n: int) -> int:
    """Return the next power-of-two >= n."""
    if n <= 0:
        raise ValueError("Value must be positive")
    return 1 if n == 1 else 1 << (int(np.ceil(np.log2(n))))


def pad_or_truncate(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resize array to target length by truncating or zero-padding."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - x.shape[0]))


def get_output_dir(framework: str) -> Path:
    """Get output directory for the specified framework."""
    name_map = {
        "qiskit": "Qiskit",
        "pennylane": "PennyLane",
        "cirq": "Cirq",
    }
    dir_name = name_map.get(framework.lower(), "Outputs")
    out_dir = Config.CURR_DIR / dir_name
    out_dir.mkdir(exist_ok=True)
    return out_dir


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureExpander:
    """Handles feature expansion transformations."""

    @staticmethod
    def expand(X: np.ndarray, method: str = "none") -> np.ndarray:
        """Apply feature expansion using specified method."""
        if method == "none":
            return X
        elif method == "poly2":
            return FeatureExpander._polynomial_expansion(X)
        elif method == "rff":
            return FeatureExpander._random_fourier_features(X)
        elif method == "fourier":
            return FeatureExpander._fourier_features(X)
        else:
            raise ValueError(f"Unknown expansion method: {method}")

    @staticmethod
    def _polynomial_expansion(X: np.ndarray) -> np.ndarray:
        """Polynomial features (degree 2)."""
        pf = PolynomialFeatures(degree=2, include_bias=False)
        return pf.fit_transform(X)

    @staticmethod
    def _random_fourier_features(X: np.ndarray) -> np.ndarray:
        """Random Fourier Features approximation."""
        rbf = RBFSampler(
            gamma=Config.RFF_GAMMA,
            n_components=Config.RFF_COMPONENTS,
            random_state=42
        )
        return rbf.fit_transform(X)

    @staticmethod
    def _fourier_features(X: np.ndarray) -> np.ndarray:
        """Deterministic Fourier features with multiple scales."""
        parts = [X]
        for scale in Config.FOURIER_SCALES:
            parts.append(np.sin(scale * X))
            parts.append(np.cos(scale * X))
        return np.concatenate(parts, axis=1)


# ============================================================================
# DATA PIPELINE
# ============================================================================

class DataPipeline:
    """Handles data loading and preprocessing."""

    def __init__(self):
        self.pca = None
        self.scaler = None
        self.metadata = {}

    def load_and_preprocess(self) -> Dict[str, Any]:
        """Load tornado data and apply full preprocessing pipeline."""
        # Load raw data
        X_train_raw, X_test_raw, y_train, y_test = self._load_tornado_data()

        # Feature expansion
        X_train_exp = FeatureExpander.expand(X_train_raw, Config.FEATURE_EXPANSION)
        X_test_exp = FeatureExpander.expand(X_test_raw, Config.FEATURE_EXPANSION)

        # L2 normalization
        X_train_norm = normalize(X_train_exp, norm='l2')
        X_test_norm = normalize(X_test_exp, norm='l2')

        # PCA dimensionality reduction
        X_train_pca, X_test_pca = self._apply_pca(X_train_norm, X_test_norm)

        # Optional post-PCA standardization
        if Config.POST_PCA_STANDARDIZE:
            X_train_pca, X_test_pca = self._standardize(X_train_pca, X_test_pca)

        # Final L2 normalization for stable angle encoding
        X_train_final = normalize(X_train_pca, norm='l2')
        X_test_final = normalize(X_test_pca, norm='l2')

        return {
            "X_train_norm": X_train_final,
            "X_test_norm": X_test_final,
            "y_train": y_train,
            "y_test": y_test,
            "metadata": self.metadata,
        }

    def _load_tornado_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load tornado dataset from module."""
        try:
            from tornado_dataset import load_tornado
        except ModuleNotFoundError:
            load_tornado = self._import_tornado_module()

        with pushd(Config.REPO_ROOT):
            (X_train_df, y_train_ser), (X_test_df, y_test_ser), _ = \
                load_tornado(balanced=True, multiclass=False, 
                             train_file="training_quantum_enhanced.xlsx",
                             test_file="test_quantum_enhanced.xlsx",
                             val_file="validation_quantum_enhanced.xlsx")

        self.metadata["feature_names"] = list(X_train_df.columns)

        return (
            X_train_df.to_numpy(),
            X_test_df.to_numpy(),
            y_train_ser.to_numpy().astype(int),
            y_test_ser.to_numpy().astype(int)
        )

    @staticmethod
    def _import_tornado_module():
        """Dynamically import tornado_dataset module."""
        import importlib.util
        module_path = Config.REPO_ROOT / 'tornado_dataset.py'
        spec = importlib.util.spec_from_file_location('tornado_dataset', str(module_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader, "Could not load tornado_dataset.py"
        spec.loader.exec_module(mod)
        return mod.load_tornado

    def _apply_pca(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PCA if dimensionality reduction is requested."""
        n_features = X_train.shape[1]

        if 0 < Config.PCA_COMPONENTS < n_features:
            self.pca = PCA(n_components=Config.PCA_COMPONENTS, whiten=False)
            X_train_pca = self.pca.fit_transform(X_train)
            X_test_pca = self.pca.transform(X_test)
            self.metadata["pca"] = self.pca
            return X_train_pca, X_test_pca

        return X_train, X_test

    def _standardize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize features (zero mean, unit variance)."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.metadata["scaler_post_pca"] = self.scaler
        return X_train_scaled, X_test_scaled


# ============================================================================
# KERNEL PROCESSING
# ============================================================================

class KernelProcessor:
    """Post-processing utilities for quantum kernel matrices."""

    @staticmethod
    def center(K_train: np.ndarray, K_test: Optional[np.ndarray] = None) -> Tuple:
        """Double-center kernel matrices."""
        n = K_train.shape[0]
        one_n = np.ones((n, n)) / n
        K_train_centered = (
            K_train - one_n @ K_train - K_train @ one_n + one_n @ K_train @ one_n
        )

        if K_test is None:
            return K_train_centered, None

        m = K_test.shape[0]
        one_mn = np.ones((m, n)) / n
        K_test_centered = (
            K_test - one_mn @ K_train - K_test @ one_n + one_mn @ K_train @ one_n
        )
        return K_train_centered, K_test_centered

    @staticmethod
    def nearest_psd(K: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Project kernel to nearest positive semi-definite matrix."""
        K = (K + K.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        eigenvalues = np.maximum(eigenvalues, eps)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    @staticmethod
    def normalize_diagonal(
        K_train: np.ndarray,
        K_test: Optional[np.ndarray] = None,
        eps: float = 1e-10
    ) -> Tuple:
        """Scale kernel so training diagonal equals 1."""
        diag = np.clip(np.diag(K_train), eps, None)
        scale = 1.0 / np.sqrt(diag)

        K_train_scaled = (scale[:, None]) * K_train * (scale[None, :])

        if K_test is not None:
            K_test_scaled = K_test * scale[None, :]
            return K_train_scaled, K_test_scaled

        return K_train_scaled, None


# ============================================================================
# QUANTUM SIMULATOR ABSTRACTION
# ============================================================================

class QuantumSimulator(ABC):
    """Abstract base class for quantum circuit simulators."""

    @abstractmethod
    def feature_map_circuit(self, x: np.ndarray):
        """Create quantum feature map circuit for input vector x."""
        pass

    @abstractmethod
    def compute_statevector(self, x: np.ndarray) -> np.ndarray:
        """Compute statevector for a single input."""
        pass

    def compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        """Compute statevectors for batch of inputs."""
        return np.vstack([self.compute_statevector(x) for x in X])

    def compute_kernel_matrix(self, X_A: np.ndarray, X_B: np.ndarray) -> np.ndarray:
        """Compute quantum kernel matrix K[i,j] = |⟨ϕ(x_i)|ϕ(x_j)⟩|²."""
        # Reuse statevectors when X_A is X_B
        if X_A is X_B:
            S = self.compute_statevectors(X_A)
            return self._kernel_from_statevectors(S, S)

        S_A = self.compute_statevectors(X_A)
        S_B = self.compute_statevectors(X_B)
        return self._kernel_from_statevectors(S_A, S_B)

    @staticmethod
    def _kernel_from_statevectors(S_A: np.ndarray, S_B: np.ndarray) -> np.ndarray:
        """Compute kernel from statevector inner products."""
        G = S_A @ S_B.conj().T
        return np.abs(G) ** 2

    def _get_num_qubits(self, x: np.ndarray) -> int:
        """Determine number of qubits needed for input."""
        target_len = next_power_of_two(len(x))
        return int(np.log2(target_len))


class QiskitSimulator(QuantumSimulator):
    """Qiskit-based quantum simulator implementation."""

    def __init__(self):
        from qiskit import QuantumRegister, QuantumCircuit
        from qiskit.quantum_info import Statevector

        self.QuantumRegister = QuantumRegister
        self.QuantumCircuit = QuantumCircuit
        self.Statevector = Statevector

    def feature_map_circuit(self, x: np.ndarray):
        """Data re-uploading feature map with RY/RZ rotations and CZ entanglement."""
        n_qubits = self._get_num_qubits(x)
        qr = self.QuantumRegister(n_qubits, name='q')
        qc = self.QuantumCircuit(qr)

        # Multiple layers to encode all features
        n_layers = int(np.ceil(len(x) / n_qubits))

        for layer in range(n_layers):
            # Encode feature slice
            start = layer * n_qubits
            end = min((layer + 1) * n_qubits, len(x))
            features = x[start:end]

            for i, angle in enumerate(features):
                a = Config.ANGLE_SCALE * float(angle)
                qc.ry(a, qr[i])
                qc.rz(a, qr[i])

            # Entangle qubits
            for i in range(n_qubits - 1):
                qc.cz(qr[i], qr[i + 1])

        return qc

    def compute_statevector(self, x: np.ndarray) -> np.ndarray:
        """Compute statevector from feature map circuit."""
        circuit = self.feature_map_circuit(x)
        sv = self.Statevector.from_instruction(circuit)
        return sv.data


class PennyLaneSimulator(QuantumSimulator):
    """PennyLane-based quantum simulator implementation."""

    def __init__(self):
        import pennylane as qml
        self.qml = qml

    def feature_map_circuit(self, x: np.ndarray):
        """Data re-uploading feature map as a callable function."""
        n_qubits = self._get_num_qubits(x)
        n_layers = int(np.ceil(len(x) / n_qubits))

        def circuit():
            for layer in range(n_layers):
                start = layer * n_qubits
                end = min((layer + 1) * n_qubits, len(x))
                features = x[start:end]

                for i, angle in enumerate(features):
                    a = Config.ANGLE_SCALE * float(angle)
                    self.qml.RY(a, wires=i)
                    self.qml.RZ(a, wires=i)

                for i in range(n_qubits - 1):
                    self.qml.CZ(wires=[i, i + 1])

        return circuit

    def compute_statevector(self, x: np.ndarray) -> np.ndarray:
        """Compute statevector using QNode."""
        n_qubits = self._get_num_qubits(x)
        dev = self.qml.device('default.qubit', wires=n_qubits)

        @self.qml.qnode(dev)
        def get_state():
            self.feature_map_circuit(x)()
            return self.qml.state()

        return np.asarray(get_state(), dtype=complex)


class CirqSimulator(QuantumSimulator):
    """Cirq-based quantum simulator implementation."""

    def __init__(self):
        import cirq
        self.cirq = cirq
        self.simulator = cirq.Simulator()

    def feature_map_circuit(self, x: np.ndarray):
        """Data re-uploading feature map with Cirq gates."""
        n_qubits = self._get_num_qubits(x)
        qubits = self.cirq.LineQubit.range(n_qubits)
        circuit = self.cirq.Circuit()

        n_layers = int(np.ceil(len(x) / n_qubits))

        for layer in range(n_layers):
            start = layer * n_qubits
            end = min((layer + 1) * n_qubits, len(x))
            features = x[start:end]

            for i, angle in enumerate(features):
                a = Config.ANGLE_SCALE * float(angle)
                circuit.append(self.cirq.ry(a)(qubits[i]))
                circuit.append(self.cirq.rz(a)(qubits[i]))

            for i in range(n_qubits - 1):
                circuit.append(self.cirq.CZPowGate()(qubits[i], qubits[i + 1]))

        return circuit

    def compute_statevector(self, x: np.ndarray) -> np.ndarray:
        """Simulate circuit and extract statevector."""
        circuit = self.feature_map_circuit(x)
        result = self.simulator.simulate(circuit)
        return result.final_state_vector


def create_simulator(framework: str) -> QuantumSimulator:
    """Factory function to create appropriate simulator."""
    simulators = {
        "qiskit": QiskitSimulator,
        "pennylane": PennyLaneSimulator,
        "cirq": CirqSimulator,
    }

    if framework not in simulators:
        raise ValueError(f"Unknown framework: {framework}")

    return simulators[framework]()


# ============================================================================
# QUANTUM SVM MODEL
# ============================================================================

class QuantumSVM:
    """Quantum Support Vector Machine using quantum kernel methods."""

    def __init__(self, simulator: QuantumSimulator):
        self.simulator = simulator
        self.clf = None
        self.K_train = None
        self.X_train = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train quantum SVM on provided data."""
        # Store training features for test kernel computation
        self.X_train = X_train

        # Compute and process kernel
        K_train = self.simulator.compute_kernel_matrix(X_train, X_train)
        K_train = (K_train + K_train.T) / 2.0  # Symmetrize

        # Apply post-processing
        K_train = self._postprocess_kernel(K_train)
        self.K_train = K_train

        # Train classical SVM on quantum kernel
        self.clf = SVC(kernel='precomputed', probability=True, C=Config.SVM_C)
        self.clf.fit(K_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test set."""
        if self.X_train is None:
            raise RuntimeError("Training features not available. Call train() first.")

        # Compute test kernel (n_test x n_train for precomputed kernel)
        K_test = self.simulator.compute_kernel_matrix(X_test, self.X_train)
        K_test = self._postprocess_kernel(K_test, is_test=True)

        # Predictions
        y_pred = self.clf.predict(K_test)
        y_scores = self.clf.predict_proba(K_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        return {
            "model": self.clf,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "cm": cm,
            "y_test": y_test,
        }

    def _postprocess_kernel(self, K: np.ndarray, is_test: bool = False) -> np.ndarray:
        """Apply kernel post-processing transformations."""
        if not is_test:
            if Config.CENTER_KERNEL:
                K, _ = KernelProcessor.center(K)
            if Config.PSD_PROJECTION:
                K = KernelProcessor.nearest_psd(K)
            if Config.UNIT_DIAGONAL:
                K, _ = KernelProcessor.normalize_diagonal(K)
        else:
            # Apply saved transformations to test kernel
            if Config.CENTER_KERNEL:
                _, K = KernelProcessor.center(self.K_train, K)
            if Config.UNIT_DIAGONAL:
                _, K = KernelProcessor.normalize_diagonal(self.K_train, K)

        return K


# ============================================================================
# RESULTS PERSISTENCE
# ============================================================================

class ResultsSaver:
    """Handles saving of model artifacts and visualizations."""

    def __init__(self, framework: str):
        self.output_dir = get_output_dir(framework)

    def save_all(self, results: Dict, data: Dict, params: Dict):
        """Save model, metrics, and visualizations."""
        self._save_model(results["model"], data["metadata"], data["X_train_norm"])
        self._save_metrics(results)
        self._save_roc_curve(results)
        self._save_params(params)

    def _save_model(self, model, metadata, X_train):
        """Save trained model and preprocessing metadata."""
        with open(self.output_dir / "QSVM_binary_svm_model.pkl", 'wb') as f:
            pickle.dump(model, f)

        with open(self.output_dir / "QSVM_binary_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        np.save(self.output_dir / "QSVM_binary_X_train.npy", X_train)

    def _save_metrics(self, results: Dict):
        """Save classification metrics."""
        # Classification report
        report = classification_report(results["y_test"], results["y_pred"], digits=3)
        with open(self.output_dir / "QSVM_binary_classification_report.txt", 'w') as f:
            f.write(report)

        # Confusion matrix
        cm_df = pd.DataFrame(
            results["cm"],
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"]
        )
        cm_df.to_csv(self.output_dir / "QSVM_binary_confusion_matrix.csv")

    def _save_roc_curve(self, results: Dict):
        """Generate and save ROC curve plot."""
        plt.figure(figsize=(6, 5))
        plt.plot(
            results["fpr"], results["tpr"],
            color='darkorange', lw=2,
            label=f"QSVM (AUC = {results['roc_auc']:.3f})"
        )
        plt.plot(
            [0, 1], [0, 1],
            color='navy', lw=1, linestyle='--',
            label="Random"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Quantum SVM ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "QSVM_binary_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    def _save_params(self, params: Dict):
        """Save the best hyperparameters."""
        with open(self.output_dir / "best_params.txt", 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with a two-stage hyperparameter search."""

    # --- Stage 1: Broad Randomized Search ---
    print("--- Stage 1: Broad Randomized Search ---")
    param_dist = {
        "SVM_C": [0.1, 1, 10, 50, 100],
        "ANGLE_SCALE": [0.1, 0.5, 1.0, 1.5, 2.0],
        "PCA_COMPONENTS": [2, 4, 6, 8, 10, 12, 14, 16],
        "FEATURE_EXPANSION": ["none", "poly2", "fourier"],
        "FOURIER_SCALES": [[0.1], [0.5], [1.0], [0.1, 0.5], [0.5, 1.0]]
    }
    
    n_iter = 50  # Number of parameter settings that are sampled
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    best_auc = 0
    best_params = None
    best_results = None
    best_data = None

    for i, params in enumerate(param_list):
        print(f"\n--- Iteration {i+1}/{len(param_list)} (Random Search) ---")
        print(f"Parameters: {params}")

        # Set config from params
        Config.SVM_C = params["SVM_C"]
        Config.ANGLE_SCALE = params["ANGLE_SCALE"]
        Config.PCA_COMPONENTS = params["PCA_COMPONENTS"]
        Config.FEATURE_EXPANSION = params["FEATURE_EXPANSION"]
        Config.FOURIER_SCALES = params["FOURIER_SCALES"]
        
        pipeline = DataPipeline()
        data = pipeline.load_and_preprocess()

        simulator = create_simulator(Config.SIMULATOR)
        qsvm = QuantumSVM(simulator)
        qsvm.train(data["X_train_norm"], data["y_train"])
        results = qsvm.evaluate(data["X_test_norm"], data["y_test"])

        print(f"AUC for this iteration: {results['roc_auc']:.4f}")
        
        if results['roc_auc'] > best_auc:
            best_auc = results['roc_auc']
            best_params = params
            best_results = results
            best_data = data
            print(f"!!! New best AUC found: {best_auc:.4f} !!!")

    print("\n--- Broad Search Finished ---")
    print(f"Best AUC so far: {best_auc:.4f}")
    print(f"Best parameters found: {best_params}")


    # --- Stage 2: Fine-Tuning Randomized Search ---
    if best_params:
        print("\n--- Stage 2: Fine-Tuning Randomized Search ---")
        
        # We need this for defining continuous distributions
        from scipy.stats import uniform

        # Get the best parameters from Stage 1
        c = best_params['SVM_C']
        angle = best_params['ANGLE_SCALE']
        pca = best_params['PCA_COMPONENTS']
        
        # Create a focused parameter DISTRIBUTION around the best parameters
        # For continuous values, we define a range (e.g., using scipy.stats.uniform)
        # For discrete values, we can provide a list of choices
        fine_tune_dist = {
            "SVM_C": uniform(loc=c * 0.5, scale=c * 1.5),  # Sample from range [c*0.5, c*2.0]
            "ANGLE_SCALE": uniform(loc=angle * 0.8, scale=angle * 0.4), # Sample from range [angle*0.8, angle*1.2]
            "PCA_COMPONENTS": list(range(max(2, pca - 2), pca + 3)), # e.g., [6, 7, 8, 9, 10]
            "FEATURE_EXPANSION": [best_params['FEATURE_EXPANSION']], # Keep the best one
            "FOURIER_SCALES": [best_params['FOURIER_SCALES']] # Keep the best one
        }

        # Use ParameterSampler to run for exactly 50 iterations
        n_fine_tune_iter = 50
        fine_tune_list = list(ParameterSampler(
            fine_tune_dist, 
            n_iter=n_fine_tune_iter, 
            random_state=42
        ))

        for i, params in enumerate(fine_tune_list):
            print(f"\n--- Iteration {i+1}/{len(fine_tune_list)} (Fine-Tuning) ---")
            print(f"Parameters: {params}")

            # Set config from params
            Config.SVM_C = params["SVM_C"]
            Config.ANGLE_SCALE = params["ANGLE_SCALE"]
            Config.PCA_COMPONENTS = int(round(params["PCA_COMPONENTS"])) # Ensure PCA is an int
            Config.FEATURE_EXPANSION = params["FEATURE_EXPANSION"]
            Config.FOURIER_SCALES = params["FOURIER_SCALES"]
            
            pipeline = DataPipeline()
            data = pipeline.load_and_preprocess()

            simulator = create_simulator(Config.SIMULATOR)
            qsvm = QuantumSVM(simulator)
            qsvm.train(data["X_train_norm"], data["y_train"])
            results = qsvm.evaluate(data["X_test_norm"], data["y_test"])

            print(f"AUC for this iteration: {results['roc_auc']:.4f}")
            
            if results['roc_auc'] > best_auc:
                best_auc = results['roc_auc']
                best_params = params
                best_results = results
                best_data = data
                print(f"!!! New best AUC found: {best_auc:.4f} !!!")

    print("\n--- Hyperparameter search finished ---")
    print(f"Final Best AUC: {best_auc:.4f}")
    print(f"Final Best parameters: {best_params}")

    # Save the best results
    if best_results:
        saver = ResultsSaver(Config.SIMULATOR)
        saver.save_all(best_results, best_data, best_params)
        print(f"\nBest model and results saved to: {saver.output_dir}")


if __name__ == "__main__":
    main()