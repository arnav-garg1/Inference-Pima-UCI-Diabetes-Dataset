import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ----------------------------
# 0. Set Device (CUDA if available)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                   header=None)
X = data.iloc[:, :-1].values   # features (first 8 columns)
y = data.iloc[:, -1].values    # target (last column)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# ----------------------------
# 2. Define Three Models with More Distinct Architectures
# ----------------------------

# Basic model: two hidden layers with ReLU activation
class DiabetesNN(nn.Module):
    def __init__(self):
        super(DiabetesNN, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Deep model: a much wider and deeper network to increase capacity
class DiabetesNNDeep(nn.Module):
    def __init__(self):
        super(DiabetesNNDeep, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Dropout model: similar to Basic but with an aggressive dropout rate to underfit
class DiabetesNNDropout(nn.Module):
    def __init__(self):
        super(DiabetesNNDropout, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)  # increased dropout rate to 0.8
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ----------------------------
# 3. Training Function
# ----------------------------
def train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=300):
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        with torch.no_grad():
            model.eval()
            # Training metrics
            train_outputs = model(X_train)
            train_preds = (train_outputs >= 0.5).float()
            train_acc = (train_preds.eq(y_train).sum().item() / y_train.size(0))
            # Testing metrics
            test_outputs = model(X_test)
            test_preds = (test_outputs >= 0.5).float()
            test_acc = (test_preds.eq(y_test).sum().item() / y_test.size(0))
            test_loss = criterion(test_outputs, y_test).item()
        
        history["train_loss"].append(loss.item())
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    return history

# ----------------------------
# 4. Initialize and Train Each Model
# ----------------------------
models = {
    "Basic": DiabetesNN().to(device),
    "Deep": DiabetesNNDeep().to(device),
    "Dropout": DiabetesNNDropout().to(device)
}

histories = {}
optimizers = {}
criterion = nn.BCELoss()
epochs = 300

for name, model in models.items():
    print(f"\nTraining {name} model:")
    optimizers[name] = optim.Adam(model.parameters(), lr=0.005)
    histories[name] = train_model(model, optimizers[name], criterion,
                                  X_train_tensor, y_train_tensor,
                                  X_test_tensor, y_test_tensor, epochs=epochs)

# ----------------------------
# 5. Compute ROC Metrics and Youden's J for Each Model
# ----------------------------
roc_data = {}
predictions = {}
with torch.no_grad():
    for name, model in models.items():
        model.eval()
        # Move predictions to CPU for numpy conversion
        y_pred_prob = model(X_test_tensor).cpu().numpy().flatten()
        predictions[name] = y_pred_prob
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc}

# ----------------------------
# 6. Plot ROC Curves (One subplot per model)
# ----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, data) in enumerate(roc_data.items()):
    ax = axs[i]
    ax.plot(data["fpr"], data["tpr"], lw=2, label=f'ROC (AUC = {data["auc"]:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC: {name}')
    ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Plot Youden's J Curves (One subplot per model)
# ----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, data) in enumerate(roc_data.items()):
    ax = axs[i]
    fpr = data["fpr"]
    tpr = data["tpr"]
    thresholds = data["thresholds"]
    youden_j = tpr - fpr  # Youden's J = sensitivity - false positive rate
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    ax.plot(thresholds, youden_j, lw=2, label=f'Optimal threshold = {optimal_threshold:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel("Youden's J")
    ax.set_title(f"Youden's J: {name}")
    ax.legend(loc='best')
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Plot Accuracy and Loss Curves (Separate subplots for each model)
# ----------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 15))
for i, (name, history) in enumerate(histories.items()):
    # Accuracy curves
    ax_acc = axs[i, 0]
    ax_acc.plot(history["train_acc"], label='Train Accuracy')
    ax_acc.plot(history["test_acc"], label='Test Accuracy', linestyle='--')
    ax_acc.set_title(f"{name} Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc='best')
    
    # Loss curves
    ax_loss = axs[i, 1]
    ax_loss.plot(history["train_loss"], label='Train Loss')
    ax_loss.plot(history["test_loss"], label='Test Loss', linestyle='--')
    ax_loss.set_title(f"{name} Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc='best')
plt.tight_layout()
plt.show()
