"""
Quantum Feature Extraction + Classical ML Comparison
Uses quantum circuits to generate features, then trains classical ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add parent directory to path for imports
CURR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR_DIR.parent))

# Import quantum simulator from binary.py
from QSVM.binary import (
    Config, DataPipeline, create_simulator,
    get_output_dir
)


class QuantumFeatureExtractor:
    """
    Extract quantum features from data using statevectors.
    Unlike quantum kernels (O(n²)), this is O(n) - computes features once per sample.
    """
    
    def __init__(self, simulator_name='cirq'):
        Config.SIMULATOR = simulator_name
        self.simulator = create_simulator(simulator_name)
        self.n_qubits = None
    
    def fit(self, X, y=None):
        """Determine quantum feature dimension from first sample."""
        sample = X[0]
        self.n_qubits = self.simulator._get_num_qubits(sample)
        return self
    
    def transform(self, X):
        """
        Transform data to quantum features.
        Returns: Original features concatenated with quantum statevector features.
        """
        print(f"Extracting quantum features from {len(X)} samples...")
        
        quantum_features = []
        for i, sample in enumerate(X):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(X)} samples")
            
            # Get quantum statevector
            statevector = self.simulator.compute_statevector(sample)
            
            # Extract features from statevector
            # Method 1: Amplitudes (real and imaginary parts)
            real_parts = np.real(statevector)
            imag_parts = np.imag(statevector)
            
            # Method 2: Probabilities (measurement probabilities)
            probs = np.abs(statevector) ** 2
            
            # Method 3: Phase information
            phases = np.angle(statevector)
            
            # Combine all quantum features
            qfeats = np.concatenate([
                real_parts[:8],      # First 8 amplitude real parts
                imag_parts[:8],      # First 8 amplitude imaginary parts
                probs[:8],           # First 8 probabilities
                phases[:4],          # First 4 phases
            ])
            
            quantum_features.append(qfeats)
        
        quantum_features = np.array(quantum_features)
        print(f"  Quantum features shape: {quantum_features.shape}")
        
        # Concatenate with original features
        return np.hstack([X, quantum_features])
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    """Train model and return evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Tornadoes caught: {cm[1,1]}/{y_test.sum()}")
    print(f"False positives: {cm[0,1]}")
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.3f}")
    
    return {
        'name': model_name,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'cm': cm
    }


def plot_roc_curves(results_list, output_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'red', 'purple']
    
    for result, color in zip(results_list, colors):
        # Interpolate for smooth curves
        mean_fpr = np.linspace(0, 1, 1000)
        tpr_interp = np.interp(mean_fpr, result['fpr'], result['tpr'])
        tpr_interp[0] = 0.0
        
        plt.plot(
            mean_fpr, tpr_interp,
            color=color,
            lw=2,
            label=f"{result['name']} (AUC = {result['auc']:.3f})"
        )
    
    # Chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Classical ML + Quantum Features', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    plt.show()


def main():
    """Main execution pipeline."""
    
    print("="*70)
    print("QUANTUM FEATURE EXTRACTION + CLASSICAL ML COMPARISON")
    print("="*70)
    
    # Configuration
    simulator_name = 'cirq'  # Change to 'qiskit' or 'pennylane' if desired
    print(f"\nUsing quantum simulator: {simulator_name.upper()}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    pipeline = DataPipeline()
    data = pipeline.load_and_preprocess()
    
    X_train = data["X_train_norm"]
    X_test = data["X_test_norm"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    print(f"Original feature dimensions: {X_train.shape[1]}")
    
    # Extract quantum features
    print("\n" + "="*70)
    print("QUANTUM FEATURE EXTRACTION")
    print("="*70)
    
    quantum_extractor = QuantumFeatureExtractor(simulator_name)
    X_train_quantum = quantum_extractor.fit_transform(X_train)
    X_test_quantum = quantum_extractor.transform(X_test)
    
    print(f"\nFinal feature dimensions: {X_train_quantum.shape[1]}")
    print(f"  Original: {X_train.shape[1]}")
    print(f"  Quantum: {X_train_quantum.shape[1] - X_train.shape[1]}")
    
    # Optional: Standardize combined features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_quantum)
    X_test_scaled = scaler.transform(X_test_quantum)
    
    # Train models
    print("\n" + "="*70)
    print("TRAINING CLASSICAL ML MODELS ON QUANTUM FEATURES")
    print("="*70)
    
    models = [
        (RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        ), "Random Forest + Quantum"),
        
        (XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=8,
            random_state=42,
            eval_metric='logloss'
        ), "XGBoost + Quantum"),
        
        (LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ), "LightGBM + Quantum"),
    ]
    
    results = []
    for model, name in models:
        result = train_and_evaluate(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            model, name
        )
        results.append(result)
    
    # Plot ROC curves
    print("\n" + "="*70)
    print("GENERATING ROC CURVE COMPARISON")
    print("="*70)
    
    output_dir = get_output_dir(simulator_name)
    roc_path = output_dir / "quantum_features_roc_comparison.png"
    plot_roc_curves(results, roc_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary_df = pd.DataFrame([
        {
            'Model': r['name'],
            'AUC': r['auc'],
            'Tornadoes Caught': f"{r['cm'][1,1]}/{y_test.sum()}",
            'False Positives': r['cm'][0,1],
            'Accuracy': (r['cm'][0,0] + r['cm'][1,1]) / r['cm'].sum()
        }
        for r in results
    ])
    
    print(summary_df.to_string(index=False))
    
    best_model = max(results, key=lambda x: x['auc'])
    print(f"\nBest Model: {best_model['name']} (AUC = {best_model['auc']:.3f})")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
