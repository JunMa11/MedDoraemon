import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfn import TabPFNClassifier
from pytabkit import RealMLP_TD_Classifier
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import os

def evaluate_classification(y_true, y_pred, y_pred_proba=None, save_dir=None):
    """
    Comprehensive evaluation of classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC curve
    save_dir : str, optional
        Directory to save plots. If None, plots will be displayed but not saved
    """
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix.png', bbox_inches='tight', dpi=300)
    
    # Plot ROC curve if probabilities are provided
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f}, Accuracy = {accuracy_score(y_true, y_pred):.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_dir:
            plt.savefig(f'{save_dir}/roc_curve.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        return {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }
    
    return {
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': cm
    }


parser = argparse.ArgumentParser(description="Tabular training script for LGE embeddings")
parser.add_argument("--config", type=str)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


feature_cols = config['feature_cols']
train_tabular = pd.read_csv(config['train_data'])
val_tabular = pd.read_csv(config['val_data'])




X_train = train_tabular[feature_cols]
y_train = train_tabular["label"]
X_test = val_tabular[feature_cols]
y_test = val_tabular["label"]

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]  # align labels if needed
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]  # align labels if needed
print(len(X_train), "training samples")
print(len(X_test), "validation samples")

if config['model'] == 'TabPFN':
    model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', ignore_pretraining_limits=True)
    save_dir = f'{config["save_dir"]}/tabpfn'
elif config['model'] == 'XGBoost':
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    save_dir = f'{config["save_dir"]}/xgb'
elif config['model'] == 'RealMLP':
    model = RealMLP_TD_Classifier(device='cuda', random_state=0, n_cv=1, n_refit=0,
                              n_epochs=256, batch_size=64, hidden_sizes=[256] * 3,
                              val_metric_name='cross_entropy',
                              use_ls=True,  # for metrics like AUC / log-loss
                              lr=0.04, verbosity=2)
    save_dir = f'{config["save_dir"]}/realmlp'
os.makedirs(save_dir, exist_ok=True)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")

results = evaluate_classification(y_test, y_pred, y_prob, save_dir=save_dir)
