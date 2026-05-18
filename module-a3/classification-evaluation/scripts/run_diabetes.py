import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, OUTPUT_DIR
from src.classifiers import train_decision_tree, train_knn, train_svm
from src.data_loader import load_dataset, preprocess_diabetes_dataset
from src.evaluation import evaluate_classifiers, generate_confusion_matrix, generate_metrics


def run_diabetes():
    filepath = DATA_DIR / 'diabetes.csv'

    print('Loading dataset...')
    df = load_dataset(filepath)

    print('Preprocessing dataset...')
    X, y = preprocess_diabetes_dataset(df)

    print('Training classifiers...')
    dt_y, dt_y_pred = train_decision_tree(X, y)
    knn_y, knn_y_pred = train_knn(X, y)
    svm_y, svm_y_pred = train_svm(X, y)

    print('Generating Confusion Matrixes...')
    target_names = ['Non-Diabetic', 'Diabetic']

    generate_confusion_matrix(
        y=dt_y,
        y_pred=dt_y_pred, 
        target_names=target_names,
        output=OUTPUT_DIR / 'dt_confusion_matrix.png'
    )
    generate_confusion_matrix(
        y=knn_y, 
        y_pred=knn_y_pred, 
        target_names=target_names,
        output=OUTPUT_DIR / 'knn_confusion_matrix.png'
    )
    generate_confusion_matrix(
        y=svm_y, 
        y_pred=svm_y_pred, 
        target_names=target_names,
        output=OUTPUT_DIR / 'svm_confusion_matrix.png'
    )
    
    print('Generating Metrics...')
    generate_metrics(dt_y, dt_y_pred, target_names)
    generate_metrics(knn_y, knn_y_pred, target_names)
    generate_metrics(svm_y, svm_y_pred, target_names)

    print('Evaluating Classifiers...')
    values = {
        'Decision Tree': (dt_y, dt_y_pred),
        'KNN': (knn_y, knn_y_pred),
        'SVM': (svm_y, svm_y_pred)
    }
    evaluate_classifiers(values)
    

if __name__ == '__main__':
    run_diabetes()