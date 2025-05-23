from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_classification(y_true, y_pred, y_pred_proba=None, save_dir=None):
    """Evaluates classification model performance with comprehensive metrics and visualizations.

    This function generates classification metrics, confusion matrix, and ROC curve (if probabilities
    are provided) for evaluating binary classification model performance. Results can be saved to
    a specified directory.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels from the model.
        y_pred_proba (array-like, optional): Predicted probabilities for ROC curve. Defaults to None.
        save_dir (str, optional): Directory path to save plots. If None, plots are only displayed.
            Defaults to None.

    Returns:
        dict: Dictionary containing evaluation metrics including:
            - classification_report (dict): Detailed classification metrics
            - confusion_matrix (ndarray): Confusion matrix
            - roc_auc (float, optional): Area under ROC curve if y_pred_proba provided

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid values.
        FileNotFoundError: If save_dir is provided but directory doesn't exist.
    """
    try:
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if y_pred_proba is not None and len(y_pred_proba) != len(y_true):
            raise ValueError("y_pred_proba must have the same length as y_true")
        if save_dir and not os.path.exists(save_dir):
            raise FileNotFoundError(f"Save directory {save_dir} does not exist")

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
        plt.show()
        
        # Plot ROC curve if probabilities are provided
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
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

    except Exception as e:
        print(f"Error in evaluate_classification: {str(e)}")
        raise