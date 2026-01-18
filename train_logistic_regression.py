import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def load_train_test_data():
    train_data = pd.read_csv('credit_data_train.csv')
    test_data = pd.read_csv('credit_data_test.csv')
    
    X_train = train_data.drop(columns=['creditworthy'])
    y_train = train_data['creditworthy']
    
    X_test = test_data.drop(columns=['creditworthy'])
    y_test = test_data['creditworthy']
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics, conf_matrix, y_pred


def print_evaluation_results(metrics, conf_matrix):
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION MODEL - EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nMODEL PERFORMANCE METRICS")
    print("-" * 60)
    print(f"Accuracy:     {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"Recall:       {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:     {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"\n                  Predicted")
    print(f"                  0         1")
    print(f"Actual    0    [{conf_matrix[0][0]:5d}]   [{conf_matrix[0][1]:5d}]")
    print(f"          1    [{conf_matrix[1][0]:5d}]   [{conf_matrix[1][1]:5d}]")
    
    true_negatives = conf_matrix[0][0]
    false_positives = conf_matrix[0][1]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]
    
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX BREAKDOWN")
    print("-" * 60)
    print(f"True Positives (TP):   {true_positives:5d}  - Correctly predicted creditworthy")
    print(f"True Negatives (TN):   {true_negatives:5d}  - Correctly predicted not creditworthy")
    print(f"False Positives (FP):  {false_positives:5d}  - Incorrectly predicted creditworthy")
    print(f"False Negatives (FN):  {false_negatives:5d}  - Incorrectly predicted not creditworthy")
    print("=" * 60 + "\n")


def main():
    print("\nLoading training and test data...")
    X_train, X_test, y_train, y_test = load_train_test_data()
    
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    print("\nTraining Logistic Regression model...")
    logistic_model = train_logistic_regression(X_train, y_train)
    
    print("Model training complete. Evaluating on test set...")
    model_metrics, conf_matrix, predictions = evaluate_model(logistic_model, X_test, y_test)
    
    print_evaluation_results(model_metrics, conf_matrix)
    
    return logistic_model, model_metrics


if __name__ == "__main__":
    trained_model, performance_metrics = main()
