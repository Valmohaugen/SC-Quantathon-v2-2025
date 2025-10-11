import os
import pickle
import sys
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added for saving confusion matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# ===================================================================================
#                                Helper Utilities
# ===================================================================================

# Minimal Config class needed for the data loading function
class Config:
    CURR_DIR = Path(__file__).resolve().parent
    REPO_ROOT = CURR_DIR.parent

# Add repo root to path for tornado_dataset import
if str(Config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(Config.REPO_ROOT))

@contextmanager
def pushd(new_dir: Path):
    """Context manager for temporary directory changes."""
    prev = os.getcwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(prev)

# ===================================================================================
#                                Data Preparation
# ===================================================================================

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
        return load_tornado(balanced=True, multiclass=False,
                            train_file="training_quantum_enhanced.xlsx",
                            test_file="test_quantum_enhanced.xlsx",
                            val_file="validation_quantum_enhanced.xlsx")

# Load data using the new function
(X_train, y_train), (X_test, y_test), (X_val, y_val) = load_tornado_data()

# Scale the data for better neural network performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===================================================================================
#                                   Classical Layers
# ===================================================================================
n_qubits = 4
n_q_layers = 4
num_qparams = n_q_layers * n_qubits * 3
n_classes = 2

class ClassicalNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, num_qparams)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return x

# ===================================================================================
#                                Quantum Circuit
# ===================================================================================
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("Using high-performance lightning.qubit device.")
except qml.DeviceError:
    dev = qml.device("default.qubit", wires=n_qubits)
    print("Using default.qubit device.")

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_nn(inputs):
    weights = inputs.view(n_q_layers, n_qubits, 3)
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.classical = ClassicalNN(input_features)
        self.quantum = qml.qnn.TorchLayer(quantum_nn, weight_shapes={})
        self.out = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = self.classical(x)
        quantum_outputs = [self.quantum(sample) for sample in x]
        x = torch.stack(quantum_outputs)
        x = self.out(x)
        return x

# ===================================================================================
#                                   Training
# ===================================================================================

# Prepare data
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
input_features = X_train.shape[1]
model = HybridModel(input_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

num_epochs = 100
patience = 20
best_val_loss = np.inf
best_state = None
counter = 0

train_losses, val_losses = [], []

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss_total = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    train_loss = train_loss_total / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss_total += criterion(val_outputs, y_val_batch).item()
    val_loss = val_loss_total / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f}")
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# ===================================================================================
#                                    Evaluation
# ===================================================================================

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    y_prob_class1 = probabilities[:, 1].cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob_class1)

    print("\n" + "="*20 + " Model Evaluation " + "="*20)
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nROC AUC Score: {auc:.4f}")

    return y_true, y_prob_class1, auc, report, cm

y_true, y_prob, auc, report, cm = evaluate_model(model, X_test_tensor, y_test_tensor)

# ===================================================================================
#                                Saving Artifacts
# ===================================================================================
# Define the output directory
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "PennyLane"
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving artifacts to: {output_dir}")

# 1. Save the model state dictionary
model_path = output_dir / "qnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"  - Model saved to {model_path}")

# 2. Save the scaler object
scaler_path = output_dir / "scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  - Scaler saved to {scaler_path}")

# 3. Save the classification report
report_path = output_dir / "classification_report.txt"
with open(report_path, 'w') as f:
    f.write("================ Model Evaluation ================\n")
    f.write(report)
    f.write("\n================================================\n")
    f.write(f"ROC AUC Score: {auc:.4f}\n")
print(f"  - Classification report saved to {report_path}")

# 4. Save the confusion matrix as a CSV
cm_df = pd.DataFrame(cm, index=["True Negative", "True Positive"], columns=["Predicted Negative", "Predicted Positive"])
cm_path = output_dir / "confusion_matrix.csv"
cm_df.to_csv(cm_path)
print(f"  - Confusion matrix saved to {cm_path}")

# 5. Save the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(train_losses, label="Training Loss")
ax1.plot(val_losses, label="Validation Loss")
ax1.set_title("Training and Validation Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

fpr, tpr, _ = roc_curve(y_true, y_prob)
ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], 'k--', label="Random Chance")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend(loc="lower right")
ax2.grid(True)

plt.tight_layout()
plot_path = output_dir / "evaluation_plots.png"
plt.savefig(plot_path, dpi=150)
print(f"  - Plots saved to {plot_path}")
plt.show()