from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_confusion_matrix(y: pd.Series, y_pred: np.ndarray, target_names: list[str], output: str) -> None:
    conf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(4, 4))

    ax = sns.heatmap(
        data=conf_matrix, 
        annot=True,
        annot_kws={"size": 14},
        cbar=False,
        cmap='Blues',
        fmt='d',
        linecolor='white',
        linewidths=0.5,
        xticklabels=target_names,
        yticklabels=target_names
    )

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.tick_params(labelsize=12)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output)


def generate_metrics(y: pd.Series, y_pred: np.ndarray, target_names: list[str]) -> None:
    print(classification_report(y, y_pred, target_names=target_names)) 


def evaluate_classifiers(reports: dict, average: str = 'weighted') -> None:
    rows = []

    for name, (y, y_pred) in reports.items():
        rows.append({
            'classifier': name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average),
            'recall': recall_score(y, y_pred, average=average),
            'f1-score': f1_score(y, y_pred, average=average)
        })
    
    df = pd.DataFrame(rows)
    print('\nClassifier Comparision\n')
    print(df.to_string(index=False))

    best_classfier = df.loc[df['f1-score'].idxmax()]
    print(f'\nBest classifier: {best_classfier['classifier']} with f1-score of {best_classfier['f1-score']}')
