# -----------------------------
# Install pykan
# -----------------------------
!pip install pykan   

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from torch.utils.data import DataLoader, TensorDataset
from kan import KAN

# -----------------------------
# Load CSV Dataset
# -----------------------------
df = pd.read_csv('/content/drive/MyDrive/dataset/SWaT_Dataset_Attack_v0.csv')


df = df.sample(3000, random_state=42)

# Preprocessing
df.drop(df.columns[0], axis=1, inplace=True)  # Drop ID or unnamed index column
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)
df.replace('A ttack', 'Attack', inplace=True)  # Correct typo

# Label Encoding
df.replace('Normal', 0, inplace=True)
df.replace('Attack', 1, inplace=True)
df['Normal/Attack'] = df['Normal/Attack'].astype(int)

# Feature & target split
x = df.drop(['Normal/Attack'], axis=1)
y = df['Normal/Attack']
X_scaled = StandardScaler().fit_transform(x)
x = pd.DataFrame(X_scaled, columns=x.columns)

# -----------------------------
# Train-test split
# -----------------------------
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)
xtr_t = torch.tensor(xtr.values, dtype=torch.float32)
xte_t = torch.tensor(xte.values, dtype=torch.float32)
ytr_t = torch.tensor(ytr.values, dtype=torch.float32).unsqueeze(1)
yte_t = torch.tensor(yte.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for batch training
train_loader = DataLoader(TensorDataset(xtr_t, ytr_t), batch_size=512, shuffle=True)

# -----------------------------
# Quantum-Inspired Feature Map
# -----------------------------
class QuantumFeatureMap(nn.Module):
    def __init__(self, input_dim):
        super(QuantumFeatureMap, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=1)

# -----------------------------
# TEKAN Model
# -----------------------------
class TEKAN_Model(nn.Module):
    def __init__(self, input_dim):
        super(TEKAN_Model, self).__init__()
        self.encoder = QuantumFeatureMap(input_dim)
        self.kan = KAN(width=[2 * input_dim, 64, 1], grid=20, k=3, device='cpu')

    def forward(self, x):
        x_enc = self.encoder(x)
        kan_out = self.kan(x_enc)

        if not torch.is_tensor(kan_out):
            kan_out = torch.from_numpy(np.asarray(kan_out)).float()

        kan_out = kan_out.to(x_enc.dtype)
        try:
            kan_out = kan_out.to(x.device)
        except Exception:
            pass

        return kan_out

# -----------------------------
# Model Training with Validation
# -----------------------------
model = TEKAN_Model(input_dim=xtr.shape[1])
trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable_params, lr=0.003) if trainable_params else torch.optim.Adam([], lr=0.003)
loss_fn = nn.BCEWithLogitsLoss()

train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

for epoch in range(100):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for xb, yb in train_loader:
        logits = model(xb)

        if not torch.is_tensor(logits):
            logits = torch.from_numpy(np.asarray(logits)).float()
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(logits).detach() > 0.5).int()
        all_preds.extend(preds.cpu().numpy().ravel().tolist())
        all_targets.extend(yb.cpu().numpy().ravel().tolist())

    avg_train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_targets, all_preds)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    with torch.no_grad():
        val_logits = model(xte_t)
        if not torch.is_tensor(val_logits):
            val_logits = torch.from_numpy(np.asarray(val_logits)).float()
        if val_logits.dim() == 1:
            val_logits = val_logits.unsqueeze(1)

        val_loss = loss_fn(val_logits, yte_t).item()
        val_preds = (torch.sigmoid(val_logits) > 0.5).int()
        val_acc = accuracy_score(yte_t.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

# -----------------------------
# Final Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    logits = model(xte_t)
    if not torch.is_tensor(logits):
        logits = torch.from_numpy(np.asarray(logits)).float()
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)

    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    ypred = (probs > 0.5).astype(int)

# Metrics
y_true = yte.to_numpy().ravel()
acc = accuracy_score(y_true, ypred)
precision = precision_score(y_true, ypred, zero_division=0)
recall = recall_score(y_true, ypred, zero_division=0)
f1 = f1_score(y_true, ypred, zero_division=0)
auc_score = roc_auc_score(y_true, probs)
rmse = np.sqrt(mean_squared_error(y_true, probs)) # Removed squared=False
mae = mean_absolute_error(y_true, probs)

print("\n Final Evaluation Metrics:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc_score:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, ypred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - TEKAN")
plt.show()

# -----------------------------
# Training & Validation Curves
# -----------------------------
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='red')
plt.plot(epochs, val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
plt.plot(epochs, val_accuracies, label='Val Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='lightgreen', lw=2, label=f'TEKAN ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - TEKAN')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# -----------------------------
# Polar Plot for RMSE & MAE
# -----------------------------
methods = ["Proposed-TEKAN", "RF", "SVM", "K+AE", "AE"]
rmse_values = [rmse, 0.27, 0.21, 0.18, 0.22]
mae_values  = [mae, 0.09, 0.07, 0.03, 0.05]

angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False).tolist()
rmse_plot = rmse_values + rmse_values[:1]
mae_plot = mae_values + mae_values[:1]
angles += angles[:1]

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, rmse_plot, linewidth=2, label="RMSE")
ax.fill(angles, rmse_plot, alpha=0.25)
ax.plot(angles, mae_plot, linewidth=2, label="MAE")
ax.fill(angles, mae_plot, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(methods, fontsize=10)
ax.set_title("Polar Plot for RMSE and MAE Performance", size=14, weight="bold")
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
plt.show()