import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def load_train_test_data():
    train_data = pd.read_csv('credit_data_train.csv')
    test_data = pd.read_csv('credit_data_test.csv')
    
    X_train = train_data.drop(columns=['creditworthy'])
    y_train = train_data['creditworthy']
    
    X_test = test_data.drop(columns=['creditworthy'])
    y_test = test_data['creditworthy']
    
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train, random_state=42):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
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
    
    return metrics, conf_matrix, y_pred, y_pred_proba


def extract_feature_importances(model, feature_names):
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    return feature_importance_df


def print_model_metrics(model_name, metrics, conf_matrix):
    print("\n" + "=" * 75)
    print(f"{model_name} - EVALUATION RESULTS")
    print("=" * 75)
    
    print("\nMODEL PERFORMANCE METRICS")
    print("-" * 75)
    print(f"Accuracy:     {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"Recall:       {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:     {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    
    print("\n" + "-" * 75)
    print("CONFUSION MATRIX")
    print("-" * 75)
    print(f"\n                  Predicted")
    print(f"                  0         1")
    print(f"Actual    0    [{conf_matrix[0][0]:5d}]   [{conf_matrix[0][1]:5d}]")
    print(f"          1    [{conf_matrix[1][0]:5d}]   [{conf_matrix[1][1]:5d}]")
    
    true_negatives = conf_matrix[0][0]
    false_positives = conf_matrix[0][1]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]
    
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"\nTN: {true_negatives}  |  FP: {false_positives}")
    print(f"FN: {false_negatives}  |  TP: {true_positives}")
    print(f"\nSensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


def print_feature_importances(model_name, feature_importance_df, top_n=10):
    print("\n" + "=" * 75)
    print(f"{model_name} - FEATURE IMPORTANCES")
    print("=" * 75)
    
    print(f"\nTOP {top_n} MOST IMPORTANT FEATURES")
    print("-" * 75)
    
    for idx, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
        bar_length = int(row['importance'] * 100)
        bar = '█' * bar_length
        print(f"{idx:2d}. {row['feature']:40s} {row['importance']:.6f}  {bar}")


def print_model_comparison(dt_metrics, rf_metrics, baseline_roc_auc=0.8473):
    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)
    
    print(f"\n{'Metric':<20} {'Decision Tree':<20} {'Random Forest':<20}")
    print("-" * 75)
    print(f"{'Accuracy':<20} {dt_metrics['accuracy']:<20.4f} {rf_metrics['accuracy']:<20.4f}")
    print(f"{'Precision':<20} {dt_metrics['precision']:<20.4f} {rf_metrics['precision']:<20.4f}")
    print(f"{'Recall':<20} {dt_metrics['recall']:<20.4f} {rf_metrics['recall']:<20.4f}")
    print(f"{'F1-Score':<20} {dt_metrics['f1_score']:<20.4f} {rf_metrics['f1_score']:<20.4f}")
    print(f"{'ROC-AUC':<20} {dt_metrics['roc_auc']:<20.4f} {rf_metrics['roc_auc']:<20.4f}")
    
    print("\n" + "=" * 75)
    print("RANDOM FOREST vs BASELINE COMPARISON")
    print("=" * 75)
    
    rf_roc_auc = rf_metrics['roc_auc']
    roc_auc_diff = rf_roc_auc - baseline_roc_auc
    roc_auc_pct_change = (roc_auc_diff / baseline_roc_auc) * 100
    
    print(f"\nBaseline (Logistic Regression) ROC-AUC:  {baseline_roc_auc:.4f}")
    print(f"Random Forest ROC-AUC:                   {rf_roc_auc:.4f}")
    print(f"Difference:                              {roc_auc_diff:+.4f}")
    print(f"Percentage Change:                       {roc_auc_pct_change:+.2f}%")
    
    if rf_roc_auc > baseline_roc_auc:
        improvement = roc_auc_diff
        print(f"\n✓ Random Forest OUTPERFORMS baseline by {improvement:.4f} ({abs(roc_auc_pct_change):.2f}%)")
    elif rf_roc_auc < baseline_roc_auc:
        underperformance = baseline_roc_auc - rf_roc_auc
        print(f"\n✗ Random Forest UNDERPERFORMS baseline by {underperformance:.4f} ({abs(roc_auc_pct_change):.2f}%)")
    else:
        print(f"\n= Random Forest matches baseline performance")


def main():
    print("=" * 75)
    print("DECISION TREE AND RANDOM FOREST CLASSIFIER TRAINING")
    print("=" * 75)
    
    print("\nLoading training and test data...")
    X_train, X_test, y_train, y_test = load_train_test_data()
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    print("\n" + "-" * 75)
    print("Training Decision Tree Classifier...")
    decision_tree_model = train_decision_tree(X_train, y_train, random_state=42)
    dt_metrics, dt_conf_matrix, dt_pred, dt_pred_proba = evaluate_model(decision_tree_model, X_test, y_test)
    print("Decision Tree training completed!")
    
    print("\nTraining Random Forest Classifier (n_estimators=100)...")
    random_forest_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    rf_metrics, rf_conf_matrix, rf_pred, rf_pred_proba = evaluate_model(random_forest_model, X_test, y_test)
    print("Random Forest training completed!")
    
    print_model_metrics("DECISION TREE CLASSIFIER", dt_metrics, dt_conf_matrix)
    print_model_metrics("RANDOM FOREST CLASSIFIER", rf_metrics, rf_conf_matrix)
    
    feature_names = X_train.columns.tolist()
    rf_feature_importance = extract_feature_importances(random_forest_model, feature_names)
    print_feature_importances("RANDOM FOREST CLASSIFIER", rf_feature_importance, top_n=10)
    
    print_model_comparison(dt_metrics, rf_metrics, baseline_roc_auc=0.8473)
    
    print("\n" + "=" * 75)
    print("TRAINING COMPLETED")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
