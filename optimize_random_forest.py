import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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


def define_parameter_grid():
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', None]
    }
    return param_dist


def optimize_random_forest(X_train, y_train, param_dist, cv_folds=5, n_iter=50):
    base_model = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting RandomizedSearchCV optimization...")
    random_search.fit(X_train, y_train)
    
    return random_search


def get_best_model_and_params(random_search):
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    return best_model, best_params, best_score


def evaluate_model_at_threshold(model, X_test, y_test, threshold):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return metrics


def test_probability_thresholds(model, X_test, y_test, thresholds):
    threshold_results = []
    
    for threshold in thresholds:
        metrics = evaluate_model_at_threshold(model, X_test, y_test, threshold)
        threshold_results.append(metrics)
    
    return threshold_results


def print_best_parameters(best_params, best_score):
    print("\n" + "=" * 80)
    print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print("\nBEST PARAMETERS FOUND")
    print("-" * 80)
    for param, value in best_params.items():
        print(f"{param:25s}: {value}")
    
    print("\n" + "-" * 80)
    print(f"Best Cross-Validation ROC-AUC Score: {best_score:.4f}")
    print("=" * 80)


def print_threshold_analysis(model, X_test, y_test, y_pred_proba_default):
    print("\n" + "=" * 80)
    print("PROBABILITY THRESHOLD TUNING ANALYSIS")
    print("=" * 80)
    
    print(f"\nTesting threshold values: 0.30 to 0.70")
    print("-" * 80)
    
    thresholds = np.arange(0.30, 0.75, 0.05)
    threshold_results = test_probability_thresholds(model, X_test, y_test, thresholds)
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FPR':<12} {'FNR':<12}")
    print("-" * 80)
    
    for result in threshold_results:
        threshold = result['threshold']
        precision = result['precision']
        recall = result['recall']
        f1 = result['f1_score']
        fpr = result['false_positive_rate']
        fnr = result['false_negative_rate']
        
        print(f"{threshold:.2f}        {precision:.4f}       {recall:.4f}       {f1:.4f}       {fpr:.4f}       {fnr:.4f}")
    
    return threshold_results


def print_detailed_threshold_metrics(threshold_results):
    print("\n" + "=" * 80)
    print("DETAILED THRESHOLD IMPACT ANALYSIS")
    print("=" * 80)
    
    for result in threshold_results:
        threshold = result['threshold']
        
        print(f"\n{'─' * 80}")
        print(f"THRESHOLD: {threshold:.2f}")
        print(f"{'─' * 80}")
        
        print(f"Accuracy:     {result['accuracy']:.4f}  ({result['accuracy']*100:.2f}%)")
        print(f"Precision:    {result['precision']:.4f}  ({result['precision']*100:.2f}%)")
        print(f"Recall:       {result['recall']:.4f}  ({result['recall']*100:.2f}%)")
        print(f"F1-Score:     {result['f1_score']:.4f}  ({result['f1_score']*100:.2f}%)")
        print(f"Sensitivity:  {result['sensitivity']:.4f}  (True Positive Rate)")
        print(f"Specificity:  {result['specificity']:.4f}  (True Negative Rate)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:   {result['tn']}")
        print(f"  False Positives:  {result['fp']}")
        print(f"  False Negatives:  {result['fn']}")
        print(f"  True Positives:   {result['tp']}")
        
        print(f"\nRisk Assessment:")
        print(f"  False Positive Rate: {result['false_positive_rate']:.4f}  (Approving bad applicants)")
        print(f"  False Negative Rate: {result['false_negative_rate']:.4f}  (Rejecting good applicants)")


def print_bank_risk_summary(threshold_results):
    print("\n" + "=" * 80)
    print("BANK RISK ASSESSMENT - THRESHOLD COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Threshold':<12} {'Approval Rate':<16} {'Bad Loan Risk':<16} {'Missed Opp.':<16} {'Recommendation':<25}")
    print("-" * 80)
    
    for result in threshold_results:
        threshold = result['threshold']
        tn = result['tn']
        fp = result['fp']
        fn = result['fn']
        tp = result['tp']
        
        total_predictions = tn + fp + fn + tp
        approval_rate = (tp + fp) / total_predictions if total_predictions > 0 else 0
        bad_loan_risk = result['false_positive_rate']
        missed_opportunity = result['false_negative_rate']
        
        if threshold <= 0.40:
            recommendation = "More Lenient"
        elif threshold >= 0.60:
            recommendation = "More Conservative"
        else:
            recommendation = "Balanced"
        
        print(f"{threshold:.2f}        {approval_rate:.2%}           {bad_loan_risk:.2%}          {missed_opportunity:.2%}          {recommendation:<25}")
    
    print("\n" + "-" * 80)
    print("GUIDANCE FOR THRESHOLD SELECTION:")
    print("-" * 80)
    print("• LOWER threshold (0.30-0.40):  Approve more applicants, higher default risk")
    print("• MEDIUM threshold (0.45-0.55): Balanced approach, typical risk tolerance")
    print("• HIGHER threshold (0.60-0.70): Conservative lending, lower default risk")


def print_optimized_model_evaluation(best_model, X_test, y_test):
    y_pred_default = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_default)
    precision = precision_score(y_test, y_pred_default)
    recall = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    conf_matrix = confusion_matrix(y_test, y_pred_default)
    
    print("\n" + "=" * 80)
    print("OPTIMIZED MODEL PERFORMANCE (Default Threshold: 0.50)")
    print("=" * 80)
    
    print("\nMODEL PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Accuracy:     {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"Precision:    {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f}  ({f1*100:.2f}%)")
    print(f"ROC-AUC:      {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
    
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print(f"\n                  Predicted")
    print(f"                  0         1")
    print(f"Actual    0    [{conf_matrix[0][0]:5d}]   [{conf_matrix[0][1]:5d}]")
    print(f"          1    [{conf_matrix[1][0]:5d}]   [{conf_matrix[1][1]:5d}]")
    
    return roc_auc


def main():
    print("=" * 80)
    print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    print("\nLoading training and test data...")
    X_train, X_test, y_train, y_test = load_train_test_data()
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    print("\n" + "-" * 80)
    print("Defining parameter grid for RandomizedSearchCV...")
    param_dist = define_parameter_grid()
    print("Parameter grid defined with 5 hyperparameters")
    
    print("\n" + "-" * 80)
    random_search = optimize_random_forest(X_train, y_train, param_dist, cv_folds=5, n_iter=50)
    
    best_model, best_params, best_score = get_best_model_and_params(random_search)
    print_best_parameters(best_params, best_score)
    
    optimized_roc_auc = print_optimized_model_evaluation(best_model, X_test, y_test)
    
    threshold_results = print_threshold_analysis(best_model, X_test, y_test, 
                                                 best_model.predict_proba(X_test)[:, 1])
    
    print_detailed_threshold_metrics(threshold_results)
    
    print_bank_risk_summary(threshold_results)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"\nFinal Optimized ROC-AUC Score: {optimized_roc_auc:.4f}\n")


if __name__ == '__main__':
    main()
