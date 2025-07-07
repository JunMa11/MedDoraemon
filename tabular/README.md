# Tabular Classification Training

A comprehensive machine learning pipeline for tabular data classification supporting multiple state-of-the-art models including TabPFN, XGBoost, and RealMLP.

## Features

- **Multiple Model Support**: Choose between TabPFN, XGBoost, and RealMLP classifiers
- **Comprehensive Evaluation**: Automatic generation of classification reports, confusion matrices, and ROC curves
- **Flexible Configuration**: YAML-based configuration for easy experiment management
- **Automated Visualization**: Saves evaluation plots and metrics to specified directories
- **GPU Support**: Automatic GPU detection and usage for compatible models

## Requirements

Install the required dependencies:

```bash
pip install xgboost scikit-learn matplotlib seaborn tabpfn pytabkit pyyaml numpy pandas torch
```

## Usage

### Basic Usage

```bash
python tabular_train.py --config config.yaml
```

### Configuration File

Create a YAML configuration file with the following structure:

```yaml
# config.yaml
model: "TabPFN"  # Options: "TabPFN", "XGBoost", "RealMLP"
train_data: "path/to/train_data.csv"
val_data: "path/to/validation_data.csv" # if set as "None", the script will automatically split the training data into training and validation sets
feature_cols: ["feature1", "feature2", "feature3"]  # List of feature column names or "all"
save_dir: "results"
```

### Data Format

Your CSV files should contain:
- Feature columns as specified in `feature_cols`
- A `label` column with binary classification labels (0/1)

Example data structure:
```csv
feature1,feature2,feature3,label
1.2,0.5,2.1,0
0.8,1.3,1.9,1
...
```

## Supported Models

### TabPFN

### XGBoost

### RealMLP


## Output

The script generates:

1. **Console Output**:
   - Training and validation sample counts
   - Accuracy score
   - Detailed classification report

2. **Saved Files** (in `save_dir/model_name/`):
   - `confusion_matrix.png`: Heatmap visualization of prediction results
   - `roc_curve.png`: ROC curve with AUC score

3. **Returned Metrics**:
   - Classification report dictionary
   - Confusion matrix
   - ROC AUC score (if applicable)

## Example Output

```
15000 training samples
3000 validation samples
Accuracy: 0.8542

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.83      0.85      1456
           1       0.85      0.88      0.86      1544

    accuracy                           0.85      3000
   macro avg       0.85      0.85      0.85      3000
weighted avg       0.85      0.85      0.85      3000
```

## File Structure

```
project/
├── tabular_train.py
├── config.yaml
├── data/
│   ├── train_data.csv
│   └── val_data.csv
└── results/
    ├── tabpfn/
    ├── xgb/
    └── realmlp/
```