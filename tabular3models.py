import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


class CFG:
    random_state = 42
    test_size = 0.2  # 20% as test
    early_stopping_rounds = 10  # catboost specific

# DataLoading from Kaggle CSV
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
X = data.iloc[:, :-1].values    # features
y = data.iloc[:, -1].values     # result y binary labels

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=CFG.test_size, random_state=CFG.random_state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=CFG.random_state)

#tabular models
models_dict = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=CFG.random_state),
    "LightGBM": lgb.LGBMClassifier(random_state=CFG.random_state),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=CFG.random_state)
}

# Train/Eval Loop
results = {}

for model_name, model in models_dict.items():
    print(f"\nTraining {model_name} model...")
    if model_name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_name == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    else:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=CFG.early_stopping_rounds) #cat
        
    # predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    loss_val = log_loss(y_test, y_pred_proba)
    results[model_name] = {"accuracy": acc, "loss": loss_val}
    print(f"{model_name} Test Accuracy: {acc:.4f}, Log Loss: {loss_val:.4f}")

#plot acc/loss for test
names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in names]
losses = [results[name]["loss"] for name in names]

#colorcode
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(names, accuracies, color=colors)
ax[0].set_title("Test Accuracy for Each Model")
ax[0].set_ylabel("Accuracy")
ax[0].set_ylim([0, 1])
ax[1].bar(names, losses, color=colors)
ax[1].set_title("Test Log Loss for Each Model")
ax[1].set_ylabel("Log Loss")
plt.show()

roc_data = {}
thresholds_dict = {}

for model_name, model in models_dict.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_val}
    thresholds_dict[model_name] = thresholds

#plot roc
fig, axs = plt.subplots(1, len(roc_data), figsize=(6 * len(roc_data), 5))
for i, model_name in enumerate(roc_data.keys()):
    axs[i].plot(roc_data[model_name]["fpr"], roc_data[model_name]["tpr"], label=f'AUC = {roc_data[model_name]["auc"]:.2f}')
    axs[i].plot([0, 1], [0, 1], 'k--')
    axs[i].set_title(f"{model_name} ROC")
    axs[i].set_xlabel("False Positive Rate")
    axs[i].set_ylabel("True Positive Rate")
    axs[i].legend(loc="lower right")
plt.tight_layout()
plt.show()

#plot youdensj
fig, axs = plt.subplots(1, len(roc_data), figsize=(6 * len(roc_data), 5))
for i, model_name in enumerate(roc_data.keys()):
    youden = roc_data[model_name]["tpr"] - roc_data[model_name]["fpr"]
    axs[i].plot(thresholds_dict[model_name], youden, label=f"Max J = {max(youden):.2f}")
    axs[i].set_title(f"{model_name} Youden's J")
    axs[i].set_xlabel("Threshold")
    axs[i].set_ylabel("Youden's J")
    axs[i].legend(loc="best")
plt.tight_layout()
plt.show()

#train data aggregation
train_results = {}
train_losses = {}
for model_name, model in models_dict.items():
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_pred_proba)
    train_results[model_name] = train_acc
    train_losses[model_name] = train_loss


x = np.arange(len(names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [train_results[name] for name in names], width, label='Train Accuracy', color='#1f77b4')
rects2 = ax.bar(x + width/2, [results[name]["accuracy"] for name in names], width, label='Test Accuracy', color='#ff7f0e')
ax.set_ylabel('Accuracy')
ax.set_title('Training vs Test Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim([0, 1])
ax.legend()
plt.show()

# loss bar chart
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [train_losses[name] for name in names], width, label='Train Log Loss', color='#2ca02c')
rects2 = ax.bar(x + width/2, [results[name]["loss"] for name in names], width, label='Test Log Loss', color='#d62728')
ax.set_ylabel('Log Loss')
ax.set_title('Training vs Test Log Loss by Model')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
plt.show()

best_model = max(results, key=lambda x: results[x]["accuracy"])
print(f"Best model based on accuracy: {best_model}") #cat
