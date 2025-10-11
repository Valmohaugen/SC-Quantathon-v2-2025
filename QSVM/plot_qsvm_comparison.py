"""
Compare QSVM (quantum kernel) vs Classical ML + Quantum Features
Plots ROC curves side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Paths
CURR_DIR = Path(__file__).resolve().parent

def load_qsvm_results(framework='Cirq'):
    """Load QSVM results from binary.py output."""
    output_dir = CURR_DIR / framework
    
    # Check if results exist
    roc_curve_path = output_dir / "binary_roc_curve.png"
    if not roc_curve_path.exists():
        print(f"⚠️  QSVM results not found in {output_dir}")
        print(f"   Run: python QSVM/binary.py first")
        return None
    
    # Load model (which contains results)
    model_path = output_dir / "binary_svm_model.pkl"
    if model_path.exists():
        print(f"✓ Found QSVM results in {framework}/")
        return True
    
    return None


def plot_comparison():
    """Plot ROC curve comparison."""
    
    print("="*70)
    print("QSVM ROC CURVE PLOTTER")
    print("="*70)
    
    # Check which frameworks have results
    frameworks = ['Cirq', 'Qiskit', 'PennyLane']
    available = []
    
    for fw in frameworks:
        if load_qsvm_results(fw):
            available.append(fw)
    
    if not available:
        print("\n❌ No QSVM results found!")
        print("\nTo generate QSVM results, run:")
        print("  python QSVM/binary.py")
        return
    
    print(f"\n✓ Found QSVM results for: {', '.join(available)}")
    
    # Display saved ROC curves
    print("\n" + "="*70)
    print("SAVED ROC CURVES:")
    print("="*70)
    
    for fw in available:
        roc_path = CURR_DIR / fw / "binary_roc_curve.png"
        print(f"\n{fw} QSVM:")
        print(f"  Location: {roc_path}")
        print(f"  View with: start {roc_path}")
    
    # Check for quantum features results
    qf_roc_path = CURR_DIR / "Cirq" / "quantum_features_roc_comparison.png"
    if qf_roc_path.exists():
        print(f"\nQuantum Features + Classical ML:")
        print(f"  Location: {qf_roc_path}")
        print(f"  View with: start {qf_roc_path}")
    else:
        print(f"\n⚠️  No quantum features results found")
        print(f"   Run: python QSVM/quantum_features_ml.py")
    
    # Create combined visualization
    print("\n" + "="*70)
    print("COMBINED VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, len(available), figsize=(6*len(available), 5))
    if len(available) == 1:
        axes = [axes]
    
    for ax, fw in zip(axes, available):
        img_path = CURR_DIR / fw / "binary_roc_curve.png"
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{fw} QSVM", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    combined_path = CURR_DIR / "qsvm_roc_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved combined plot: {combined_path}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
