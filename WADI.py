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
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error
)
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Load CSV Dataset
# -----------------------------
df = pd.read_csv('/content/drive/MyDrive/datasets/WADI_attackdataLABLE.csv', low_memory=False)

# Remove the first row (extra header)
df = df.iloc[1:].copy()
df.reset_index(drop=True, inplace=True)

label_column = '130'  # identified as label column

# Sample smaller dataset if large
df = df.sample(3000, random_state=42).copy()

# Drop first column (index-like)
df.drop(df.columns[0], axis=1, inplace=True)

df.replace('', np.nan, inplace=True)
df.dropna(subset=[label_column], inplace=True)

# Clean label column
df[label_column] = df[label_column].astype(str).str.strip()
df.replace('Attack_LABLE', 1, inplace=True)
df.replace('1', 1, inplace=True)
df.replace('-1', 0, inplace=True)
df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
df.dropna(subset=[label_column], inplace=True)
df[label_column] = df[label_column].astype(int)

# Split features & labels
x = df.drop([label_column], axis=1)
y = df[label_column]

# Convert features to numeric
for col in x.columns:
    x[col] = pd.to_numeric(x[col], errors='coerce')
x.dropna(axis=1, how='all', inplace=True)
x.fillna(0, inplace=True)

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)

# -----------------------------
# Train-test split
# -----------------------------
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
xtr_t = torch.tensor(xtr.values, dtype=torch.float32)
xte_t = torch.tensor(xte.values, dtype=torch.float32)
ytr_t = torch.tensor(ytr.values, dtype=torch.float32).unsqueeze(1)
yte_t = torch.tensor(yte.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(xtr_t, ytr_t), batch_size=128, shuffle=True)

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
# TEKAN Model with slight improvements
# -----------------------------
class TEKAN_Model(nn.Module):
    def __init__(self, input_dim):
        super(TEKAN_Model, self).__init__()
        self.encoder = QuantumFeatureMap(input_dim)
        self.kan_layers = nn.Sequential(
            nn.Linear(2 * input_dim, 128),  # slightly increased neurons
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),  # slightly reduced dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
            nn.Sigmoid()  # probability output
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.kan_layers(x)

# -----------------------------
# Training Setup
# -----------------------------
model = TEKAN_Model(input_dim=xtr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
loss_fn = nn.BCELoss()  # Because output is sigmoid

train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

def exponential_moving_average(data, alpha=0.9):
    smoothed = []
    for i, point in enumerate(data):
        if i == 0:
            smoothed.append(point)
        else:
            smoothed.append(alpha * smoothed[i-1] + (1 - alpha) * point)
    return smoothed

# Early stopping
best_val_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(150):
    model.train()
    total_loss, all_preds, all_targets = 0, [], []
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item()
        preds = logits.detach() > 0.5
        all_preds.extend(preds.int().numpy())
        all_targets.extend(yb.int().numpy())

    avg_train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_targets, all_preds)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(xte_t)
        val_loss = loss_fn(val_logits, yte_t).item()
        val_preds = (val_logits > 0.5).int()
        val_acc = accuracy_score(yte_t.numpy(), val_preds.numpy())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

model.load_state_dict(torch.load('best_model.pth'))

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    preds = model(xte_t)
    ypred = (preds > 0.5).int().numpy()

acc = accuracy_score(yte, ypred)
f1 = f1_score(yte, ypred)
precision = precision_score(yte, ypred)
recall = recall_score(yte, ypred)
auc = roc_auc_score(yte, preds.numpy())
rmse = np.sqrt(mean_squared_error(yte, preds.numpy()))
mae = mean_absolute_error(yte, preds.numpy())

print(f"\n Final Test Metrics:")
print(f"Accuracy  : {acc:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")

# -----------------------------
# Combined Plots
# -----------------------------
epochs = range(1, len(train_losses) + 1)
smooth_train_losses = exponential_moving_average(train_losses, alpha=0.9)
smooth_val_losses = exponential_moving_average(val_losses, alpha=0.9)
smooth_train_accuracies = exponential_moving_average(train_accuracies, alpha=0.9)
smooth_val_accuracies = exponential_moving_average(val_accuracies, alpha=0.9)

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Loss Curve
axs[0, 0].plot(epochs, smooth_train_losses, label='Train Loss', color='red', linewidth=2)
axs[0, 0].plot(epochs, smooth_val_losses, label='Val Loss', color='orange', linewidth=2)
axs[0, 0].set_title("Loss Over Epochs")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Accuracy Curve
axs[0, 1].plot(epochs, smooth_train_accuracies, label='Train Accuracy', color='green', linewidth=2)
axs[0, 1].plot(epochs, smooth_val_accuracies, label='Val Accuracy', color='blue', linewidth=2)
axs[0, 1].set_title("Accuracy Over Epochs")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()

# Confusion Matrix
cm = confusion_matrix(yte, ypred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=axs[1, 0], colorbar=False)
axs[1, 0].set_title("Confusion Matrix - TEKAN")

# ROC Curve
fpr, tpr, thresholds = roc_curve(yte, preds.numpy())
axs[1, 1].plot(fpr, tpr, color='green', lw=2, label=f"ROC (AUC = {auc:.2f})")
axs[1, 1].plot([0, 1], [0, 1], 'r--')
axs[1, 1].set_title("ROC Curve - TEKAN")
axs[1, 1].set_xlabel("False Positive Rate")
axs[1, 1].set_ylabel("True Positive Rate")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Polar Plot for RMSE, MAE & 1-AUC
# -----------------------------
metrics = [rmse, mae, 1-auc]
labels = ['RMSE', 'MAE', '1-AUC']

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
metrics += metrics[:1]
angles += angles[:1]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, metrics, 'o-', linewidth=2)
ax.fill(angles, metrics, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Polar Plot - RMSE, MAE, 1-AUC (TEKAN)")
ax.grid(True)
plt.show()