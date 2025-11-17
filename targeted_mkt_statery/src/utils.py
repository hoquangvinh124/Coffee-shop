"""
Utility Functions for Targeted Marketing Strategy Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', 
                          cmap=plt.cm.Blues, figsize=(10, 8)):
    """
    Vẽ confusion matrix với matplotlib
    
    Parameters:
    -----------
    cm : array
        Confusion matrix
    classes : list
        Tên các classes
    normalize : bool
        Có normalize hay không
    title : str
        Tiêu đề biểu đồ
    cmap : colormap
        Color map cho heatmap
    figsize : tuple
        Kích thước figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return plt


def evaluate_model(y_true, y_pred, class_names=None):
    """
    Đánh giá model với các metrics
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    class_names : list
        Tên các classes
        
    Returns:
    --------
    dict : Dictionary chứa các metrics
    """
    # Get unique labels from data
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Adjust class_names to match actual labels
    actual_class_names = [class_names[i] for i in labels if i < len(class_names)]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, 
                                   target_names=actual_class_names, 
                                   output_dict=True, zero_division=0)
    
    # F1 scores
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    results = {
        'confusion_matrix': cm,
        'classification_report': report,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    return results


def print_model_evaluation(results, model_name="Model"):
    """
    In kết quả đánh giá model
    
    Parameters:
    -----------
    results : dict
        Dictionary từ evaluate_model()
    model_name : str
        Tên model
    """
    print("=" * 80)
    print(f"{model_name} EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nF1-Score (Micro):    {results['f1_micro']:.4f}")
    print(f"F1-Score (Macro):    {results['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    report = results['classification_report']
    for class_name in report.keys():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[class_name]
            print(f"{class_name:20s} | Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    
    print("=" * 80)


def plot_training_history(history, figsize=(14, 5)):
    """
    Vẽ training history cho DNN
    
    Parameters:
    -----------
    history : keras.callbacks.History
        History object từ model.fit()
    figsize : tuple
        Kích thước figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=10, figsize=(10, 6)):
    """
    Vẽ feature importance
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame với columns ['feature', 'importance']
    top_n : int
        Số features top để hiển thị
    figsize : tuple
        Kích thước figure
    """
    top_features = importance_df.nlargest(top_n, 'importance')
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'], 
             color='skyblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return plt


def save_metrics_to_csv(results, model_name, save_path):
    """
    Lưu metrics vào CSV file
    
    Parameters:
    -----------
    results : dict
        Dictionary từ evaluate_model()
    model_name : str
        Tên model
    save_path : str
        Đường dẫn lưu file
    """
    report = results['classification_report']
    
    # Tạo DataFrame
    metrics_data = []
    for class_name in report.keys():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[class_name]
            metrics_data.append({
                'model': model_name,
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(save_path, index=False)
    print(f"✓ Đã lưu metrics vào: {save_path}")


def compare_models(results_dict, metric='f1_micro'):
    """
    So sánh performance của nhiều models
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary với {model_name: results}
    metric : str
        Metric để so sánh
        
    Returns:
    --------
    DataFrame : Bảng so sánh
    """
    comparison = []
    for model_name, results in results_dict.items():
        comparison.append({
            'Model': model_name,
            'F1 (Micro)': results['f1_micro'],
            'F1 (Macro)': results['f1_macro'],
            'F1 (Weighted)': results['f1_weighted']
        })
    
    df = pd.DataFrame(comparison)
    # Map metric name to column name
    metric_map = {
        'f1_micro': 'F1 (Micro)',
        'f1_macro': 'F1 (Macro)',
        'f1_weighted': 'F1 (Weighted)'
    }
    sort_col = metric_map.get(metric, 'F1 (Micro)')
    df = df.sort_values(by=sort_col, ascending=False)
    return df


def plot_model_comparison(comparison_df, figsize=(12, 6)):
    """
    Vẽ biểu đồ so sánh models
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame từ compare_models()
    figsize : tuple
        Kích thước figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(comparison_df))
    width = 0.25
    
    ax.bar(x - width, comparison_df['F1 (Micro)'], width, label='F1 (Micro)', 
           color='skyblue', edgecolor='black')
    ax.bar(x, comparison_df['F1 (Macro)'], width, label='F1 (Macro)', 
           color='lightgreen', edgecolor='black')
    ax.bar(x + width, comparison_df['F1 (Weighted)'], width, label='F1 (Weighted)', 
           color='salmon', edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("Utils module loaded successfully!")
