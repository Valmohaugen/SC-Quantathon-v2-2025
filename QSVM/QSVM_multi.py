import os
import pickle
import sys
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, KernelCenterer, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform


# Silence noisy sklearn warning
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*does not have valid feature names.*",
    category=UserWarning,
)

# Silence the Cirq deprecation warning for StatePreparationChannel
warnings.filterwarnings("ignore", category=DeprecationWarning, module='cirq.*')

# ADD THIS LINE: Silence the Cirq norm warning
warnings.filterwarnings("ignore", message=r".*final state vector's norm.*", category=UserWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

def _parse_fourier_scales(raw: str) -> list:
    """Parse comma-separated Fourier scales from environment variable."""
    if not raw:
        return [1.0]
    try:
        return [float(s) for s in raw.split(",") if s.strip()]
    except Exception:
        return [1.0]

class Config:
    """Central configuration for QSVM experiments."""
    SIMULATOR = os.getenv("QSVM_SIMULATOR", "cirq").lower()
    PCA_COMPONENTS = int(os.getenv("QSVM_PCA_COMPONENTS", "8"))
    PHASE_GAMMA = float(os.getenv("QSVM_PHASE_GAMMA", "2.0"))
    SVM_C = float(os.getenv("QSVM_C", "1.0"))

    # Feature expansion settings
    FEATURE_EXPANSION = os.getenv("QSVM_FEATURE_EXPAND", "none").lower()
    FOURIER_SCALES = _parse_fourier_scales(os.getenv("QSVM_FOURIER_SCALES", "1.0"))

    # Setup paths
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

def next_pow2_length(n: int) -> int:
    """Return the next power-of-two length >= n."""
    if n <= 0:
        raise ValueError("Vector length must be positive")
    return 1 if n == 1 else 1 << int(np.ceil(np.log2(n)))

def get_output_directory(framework: str) -> Path:
    """Get and create output directory for the specified framework."""
    mapping = {
        "qiskit": "Qiskit",
        "pennylane": "PennyLane",
        "cirq": "Cirq",
    }
    name = mapping.get(framework.lower(), "Outputs")
    out_dir = (Config.CURR_DIR / name).resolve()
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureExpander:
    """Handles feature expansion transformations."""

    @staticmethod
    def expand(X: np.ndarray, method: str) -> np.ndarray:
        if method == "none":
            return X
        elif method == "poly2":
            return PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        elif method == "fourier":
            parts = [X]
            for scale in Config.FOURIER_SCALES:
                parts.append(np.sin(scale * X))
                parts.append(np.cos(scale * X))
            return np.concatenate(parts, axis=1)
        else:
            raise ValueError(f"Unknown expansion method: {method}")

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_tornado_data():
    """Load tornado dataset dynamically."""
    try:
        from tornado_dataset import load_tornado
    except ModuleNotFoundError:
        import importlib.util
        module_path = Config.REPO_ROOT / 'tornado_dataset.py'
        spec = importlib.util.spec_from_file_location('tornado_dataset', str(module_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        load_tornado = mod.load_tornado

    with pushd(Config.REPO_ROOT):
        return load_tornado(balanced=True, multiclass=True,
                             train_file="training_quantum_enhanced.xlsx",
                             test_file="test_quantum_enhanced.xlsx",
                             val_file="validation_quantum_enhanced.xlsx")

def preprocess_data(X_train_raw, X_test_raw, y_train, y_test):
    """Apply feature expansion, standardization, PCA, and L2 normalization."""
    # 1. Feature Expansion
    X_train_exp = FeatureExpander.expand(X_train_raw, Config.FEATURE_EXPANSION)
    X_test_exp = FeatureExpander.expand(X_test_raw, Config.FEATURE_EXPANSION)

    # 2. Standardization
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train_exp)
    X_test_std = scaler.transform(X_test_exp)

    # 3. PCA Dimensionality Reduction
    # Adjust PCA components if expansion created fewer features than requested
    n_features = X_train_std.shape[1]
    n_components = min(Config.PCA_COMPONENTS, n_features)
    if n_components <= 0:
        raise ValueError(f"Number of PCA components must be positive. Got {n_components} from min({Config.PCA_COMPONENTS}, {n_features})")

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # 4. Final L2 Normalization
    X_train_norm = normalize(X_train_pca, norm='l2')
    X_test_norm = normalize(X_test_pca, norm='l2')

    return X_train_norm, X_test_norm, y_train, y_test, pca, scaler

def load_and_preprocess_data():
    """Load and preprocess tornado dataset."""
    (X_train_df, y_train_ser), (X_test_df, y_test_ser), _ = load_tornado_data()

    X_train_raw = X_train_df.to_numpy()
    X_test_raw = X_test_df.to_numpy()
    y_train = y_train_ser.to_numpy().astype(int)
    y_test = y_test_ser.to_numpy().astype(int)

    X_train_norm, X_test_norm, y_train, y_test, pca, scaler = preprocess_data(
        X_train_raw, X_test_raw, y_train, y_test
    )

    return {
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": {
            "scaler": scaler,
            "pca": pca,
            "feature_names": list(X_train_df.columns),
            "label_names": sorted(list(set(y_train))),
        },
    }

# ============================================================================
# QUANTUM SIMULATOR ABSTRACTION
# ============================================================================
class QuantumSimulator(ABC):
    """Abstract base class for quantum simulators."""
    def __init__(self):
        self.gamma = 1.0

    def set_gamma(self, gamma: float):
        self.gamma = float(gamma)

    @abstractmethod
    def feature_map_circuit(self, x: np.ndarray):
        pass

    @abstractmethod
    def statevectors_from_features(self, X: np.ndarray) -> np.ndarray:
        pass

class QiskitSimulator(QuantumSimulator):
    """Qiskit-based quantum simulator."""
    def __init__(self):
        super().__init__()
        from qiskit import QuantumRegister, QuantumCircuit
        from qiskit.circuit.library import StatePreparation as StatePrep, Diagonal as DiagonalGate
        from qiskit.quantum_info import Statevector
        self.QuantumRegister = QuantumRegister
        self.QuantumCircuit = QuantumCircuit
        self.StatePrep = StatePrep
        self.DiagonalGate = DiagonalGate
        self.Statevector = Statevector

    def feature_map_circuit(self, x: np.ndarray):
        n_features = len(x)
        # MODIFIED: Robust qubit calculation
        n_qubits = max(1, (n_features - 1).bit_length()) if n_features > 0 else 1
        target_len = 1 << n_qubits

        x_padded = np.pad(x, (0, target_len - n_features))
        norm = np.linalg.norm(x_padded)
        if not np.isfinite(norm) or np.isclose(norm, 0.0):
            amps = np.zeros(target_len)
            amps[0] = 1.0 # Default to |0> state
        else:
            amps = x_padded / norm
        
        phases = self.gamma * x_padded
        diag_elements = np.exp(1j * phases)
        
        qr = self.QuantumRegister(n_qubits, name='q')
        qc = self.QuantumCircuit(qr)
        qc.append(self.StatePrep(amps), qr)
        qc.append(self.DiagonalGate(diag_elements), qr)
        return qc

    def statevectors_from_features(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self.Statevector.from_instruction(self.feature_map_circuit(x)).data for x in X])

class PennyLaneSimulator(QuantumSimulator):
    """PennyLane-based quantum simulator."""
    def __init__(self):
        super().__init__()
        import pennylane as qml
        self.qml = qml

    def feature_map_circuit(self, x: np.ndarray):
        n_features = len(x)
        # MODIFIED: Robust qubit calculation
        n_qubits = max(1, (n_features - 1).bit_length()) if n_features > 0 else 1
        target_len = 1 << n_qubits
        
        def circuit():
            x_padded = np.pad(x, (0, target_len - n_features))
            norm = np.linalg.norm(x_padded)
            if not np.isfinite(norm) or np.isclose(norm, 0.0):
                amps = np.zeros(target_len)
                amps[0] = 1.0
            else:
                amps = x_padded / norm
            self.qml.AmplitudeEmbedding(amps, wires=range(n_qubits), normalize=False)
            phases = self.gamma * x_padded
            self.qml.DiagonalQubitUnitary(np.exp(1j * phases), wires=range(n_qubits))
        return circuit

    def statevectors_from_features(self, X: np.ndarray) -> np.ndarray:
        # NOTE: This method runs a full quantum simulation for each data point (row) in X.
        # This can be computationally expensive for large datasets.
        n_features = X.shape[1]
        n_qubits = max(1, (n_features - 1).bit_length()) if n_features > 0 else 1
        
        dev = self.qml.device('default.qubit', wires=n_qubits)
        @self.qml.qnode(dev)
        def get_state(x):
            self.feature_map_circuit(x)()
            return self.qml.state()
        return np.array([get_state(x) for x in X])

class CirqSimulator(QuantumSimulator):
    """Cirq-based quantum simulator."""
    def __init__(self):
        super().__init__()
        import cirq
        self.cirq = cirq
        self.simulator = cirq.Simulator()

    def feature_map_circuit(self, x: np.ndarray):
        n_features = len(x)
        n_qubits = max(1, (n_features - 1).bit_length()) if n_features > 0 else 1
        target_len = 1 << n_qubits

        qubits = self.cirq.LineQubit.range(n_qubits)
        circuit = self.cirq.Circuit()
        
        x_padded = np.pad(x, (0, target_len - n_features))
        norm = np.linalg.norm(x_padded)
        
        if not np.isfinite(norm) or np.isclose(norm, 0.0):
            return circuit # Return empty circuit for zero vector, results in |0> state
        
        amps = x_padded / norm
        
        # MODIFIED: Reverted to the older name compatible with your Cirq v1.6.1
        circuit.append(self.cirq.StatePreparationChannel(amps).on(*qubits))
        
        phases = self.gamma * x_padded
        circuit.append(self.cirq.DiagonalGate(np.exp(1j * phases)).on(*qubits))
        return circuit

    def statevectors_from_features(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self.simulator.simulate(self.feature_map_circuit(x)).final_state_vector for x in X])

def create_simulator(framework: str) -> QuantumSimulator:
    simulators = {"qiskit": QiskitSimulator, "pennylane": PennyLaneSimulator, "cirq": CirqSimulator}
    sim_class = simulators.get(framework.lower())
    if not sim_class: raise ValueError(f"Unknown framework: {framework}")
    return sim_class()

# ============================================================================
# KERNEL & SVM
# ============================================================================
def build_kernel_matrix(sim: QuantumSimulator, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    S_A = sim.statevectors_from_features(XA)
    S_B = sim.statevectors_from_features(XB)
    G = S_A @ S_B.conj().T
    return np.abs(G) ** 2

def build_roc_results(y_test, y_scores, classes):
    Y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y_test_bin[:, i], y_scores[:, i])
        fpr_dict[classes[i]] = fpr
        tpr_dict[classes[i]] = tpr
        roc_auc_dict[classes[i]] = auc(fpr, tpr)
    fpr_micro, tpr_micro, _ = roc_curve(Y_test_bin.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_dict.values()]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in classes:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    return {
        "roc_auc_dict": roc_auc_dict, "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": roc_auc_macro, "fpr_dict": fpr_dict, "tpr_dict": tpr_dict,
        "fpr_micro": fpr_micro, "tpr_micro": tpr_micro, "fpr_macro": all_fpr, "tpr_macro": mean_tpr,
    }

def train_and_evaluate_svm(data, simulator: QuantumSimulator):
    X_train, y_train = data["X_train_norm"], data["y_train"]
    X_test, y_test = data["X_test_norm"], data["y_test"]
    K_train = build_kernel_matrix(simulator, X_train, X_train)
    K_test = build_kernel_matrix(simulator, X_test, X_train)
    kc = KernelCenterer()
    K_train_centered = kc.fit_transform(K_train)
    K_test_centered = kc.transform(K_test)
    clf = SVC(kernel='precomputed', probability=True, C=Config.SVM_C, class_weight='balanced')
    clf.fit(K_train_centered, y_train)
    y_pred = clf.predict(K_test_centered)
    y_scores = clf.predict_proba(K_test_centered)
    classes = sorted(list(set(y_train)))
    roc_metrics = build_roc_results(y_test, y_scores, classes)
    return {
        "model": clf, "y_pred": y_pred, "y_scores": y_scores, "y_test": y_test,
        "accuracy": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "classes": classes, **roc_metrics
    }
# ============================================================================
# VISUALIZATION & SAVING
# ============================================================================
def plot_roc_curve(results, out_dir):
    """
    Generate and save a detailed ROC curve visualization for multiclass results,
    showing individual class curves along with micro and macro averages.
    """
    plt.figure(figsize=(9, 8))
    plt.plot(
        results["fpr_micro"], results["tpr_micro"],
        label=f"Micro-average (AUC = {results['roc_auc_micro']:.3f})",
        color="deeppink", linestyle=":", linewidth=4,
    )
    plt.plot(
        results["fpr_macro"], results["tpr_macro"],
        label=f"Macro-average (AUC = {results['roc_auc_macro']:.3f})",
        color="navy", linestyle=":", linewidth=4,
    )
    colors = ["aqua", "darkorange", "cornflowerblue", "limegreen"]
    for i, color in zip(results["classes"], colors):
        if i in results["fpr_dict"]:
            plt.plot(
                results["fpr_dict"][i], results["tpr_dict"][i], color=color, lw=2,
                label=f"Class {i} (AUC = {results['roc_auc_dict'][i]:.3f})",
            )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multiclass Quantum SVM ROC Curve", fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "QSVM_multi_roc_curve.png", dpi=150)
    plt.close()

def save_artifacts(results, data, params, framework: str):
    """Save model, metrics, and visualizations."""
    out_dir = get_output_directory(framework)
    with open(out_dir / "QSVM_multi_svm_model.pkl", 'wb') as f: pickle.dump(results["model"], f)
    with open(out_dir / "QSVM_multi_metadata.pkl", 'wb') as f: pickle.dump(data["metadata"], f)
    with open(out_dir / "QSVM_multi_best_params.txt", 'w') as f:
        for key, value in params.items(): f.write(f"{key}: {value}\n")
    report = classification_report(results["y_test"], results["y_pred"], digits=3)
    with open(out_dir / "QSVM_multi_classification_report.txt", 'w') as f: f.write(report)
    cm_df = pd.DataFrame(results["cm"], index=[f"true_{l}" for l in results['classes']], columns=[f"pred_{l}" for l in results['classes']])
    cm_df.to_csv(out_dir / "QSVM_multi_confusion_matrix.csv")
    plot_roc_curve(results, out_dir)

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution pipeline with two-stage hyperparameter search."""
    best_geo_mean = 0.0
    best_params_overall, best_results_overall, best_data_overall = None, None, None

    # --- Stage 1: Broad Randomized Search ---
    print("--- Stage 1: Broad Randomized Search ---")
    param_dist = {
        "SVM_C": [1, 10, 50, 100],
        "PHASE_GAMMA": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "PCA_COMPONENTS": [4, 6, 8],
        "FEATURE_EXPANSION": ["fourier", "none"],
        "FOURIER_SCALES": [[0.5], [1.0], [1.5], [0.5, 1.0], [1.0, 2.0]]
    }
    n_iter = 50
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    for i, params in enumerate(param_list):
        print(f"\n--- Iteration {i+1}/{len(param_list)} (Random Search) ---")
        print(f"Parameters: {params}")

        # Set config from params
        for key, value in params.items():
            setattr(Config, key.upper(), value)

        data = load_and_preprocess_data()
        simulator = create_simulator(Config.SIMULATOR)
        simulator.set_gamma(Config.PHASE_GAMMA)
        results = train_and_evaluate_svm(data, simulator)

        all_aucs = list(results['roc_auc_dict'].values()) + [results['roc_auc_micro'], results['roc_auc_macro']]
        current_geo_mean = np.prod([auc + 1e-9 for auc in all_aucs]) ** (1.0 / len(all_aucs))
        print(f"AUC Geometric Mean for this iteration: {current_geo_mean:.4f}")

        if current_geo_mean > best_geo_mean:
            best_geo_mean, best_params_overall, best_results_overall, best_data_overall = \
                current_geo_mean, params, results, data
            print(f"!!! New best overall performance found (AUC Geo Mean: {best_geo_mean:.4f}) !!!")

    print("\n--- Broad Search Finished ---")
    if best_params_overall:
        print(f"Best AUC Geometric Mean so far: {best_geo_mean:.4f}")
        print(f"Achieved with parameters: {best_params_overall}")

    # --- Stage 2: Fine-Tuning Randomized Search ---
    if best_params_overall:
        print("\n--- Stage 2: Fine-Tuning Randomized Search ---")
        c = best_params_overall['SVM_C']
        gamma = best_params_overall['PHASE_GAMMA']
        pca = best_params_overall['PCA_COMPONENTS']

        fine_tune_dist = {
            "SVM_C": uniform(loc=c * 0.5, scale=c),
            "PHASE_GAMMA": uniform(loc=gamma * 0.8, scale=gamma * 0.4),
            "PCA_COMPONENTS": list(range(max(2, pca - 2), pca + 3)),
            "FEATURE_EXPANSION": [best_params_overall['FEATURE_EXPANSION']],
            "FOURIER_SCALES": [best_params_overall['FOURIER_SCALES']]
        }
        # MODIFIED: Changed number of iterations as requested
        n_fine_tune_iter = 50
        fine_tune_list = list(ParameterSampler(fine_tune_dist, n_iter=n_fine_tune_iter, random_state=42))

        for i, params in enumerate(fine_tune_list):
            print(f"\n--- Iteration {i+1}/{len(fine_tune_list)} (Fine-Tuning) ---")
            print(f"Parameters: {params}")

            for key, value in params.items():
                setattr(Config, key.upper(), value)
            Config.PCA_COMPONENTS = int(round(Config.PCA_COMPONENTS))

            data = load_and_preprocess_data()
            simulator = create_simulator(Config.SIMULATOR)
            simulator.set_gamma(Config.PHASE_GAMMA)
            results = train_and_evaluate_svm(data, simulator)

            all_aucs = list(results['roc_auc_dict'].values()) + [results['roc_auc_micro'], results['roc_auc_macro']]
            current_geo_mean = np.prod([auc + 1e-9 for auc in all_aucs]) ** (1.0 / len(all_aucs))
            print(f"AUC Geometric Mean for this iteration: {current_geo_mean:.4f}")

            if current_geo_mean > best_geo_mean:
                best_geo_mean, best_params_overall, best_results_overall, best_data_overall = \
                    current_geo_mean, params, results, data
                print(f"!!! New best overall performance found (AUC Geo Mean: {best_geo_mean:.4f}) !!!")

    # --- Final Summary ---
    print("\n--- Hyperparameter search finished ---")
    if best_params_overall:
        print(f"\n--- Best Overall Performance Summary ---")
        print(f"Highest Geometric Mean of AUCs (6 metrics): {best_geo_mean:.4f}")
        print(f"Best Hyperparameters: {best_params_overall}")
        print("\nIndividual AUC scores from this single best run:")
        for class_label, auc_score in best_results_overall['roc_auc_dict'].items():
            print(f"  - Class {class_label} AUC: {auc_score:.4f}")
        print(f"  - Micro Avg AUC: {best_results_overall['roc_auc_micro']:.4f}")
        print(f"  - Macro Avg AUC: {best_results_overall['roc_auc_macro']:.4f}")
        save_artifacts(best_results_overall, best_data_overall, best_params_overall, Config.SIMULATOR)
        print(f"\nModel artifacts from the best overall run saved to: {get_output_directory(Config.SIMULATOR)}")
    else:
        print("No successful runs completed. No artifacts to save.")

if __name__ == "__main__":
    main()