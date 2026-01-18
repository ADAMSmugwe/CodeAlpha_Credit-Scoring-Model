import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def load_train_test_data():
    train_data = pd.read_csv('credit_data_train.csv')
    test_data = pd.read_csv('credit_data_test.csv')
    
    X_train = train_data.drop(columns=['creditworthy'])
    y_train = train_data['creditworthy']
    
    X_test = test_data.drop(columns=['creditworthy'])
    y_test = test_data['creditworthy']
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def extract_feature_importance(model, feature_names):
    coefficients = model.coef_[0]
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    feature_importance_df = feature_importance_df.sort_values('coefficient', ascending=False)
    
    return feature_importance_df


def analyze_coefficient_logic(feature_importance_df):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    print("\nTOP 5 POSITIVE COEFFICIENTS (Increase Creditworthiness)")
    print("-" * 70)
    top_positive = feature_importance_df.head(5)
    for idx, row in top_positive.iterrows():
        print(f"{row['feature']:30s}  {row['coefficient']:+.4f}")
    
    print("\nTOP 5 NEGATIVE COEFFICIENTS (Decrease Creditworthiness)")
    print("-" * 70)
    top_negative = feature_importance_df.tail(5).iloc[::-1]
    for idx, row in top_negative.iterrows():
        print(f"{row['feature']:30s}  {row['coefficient']:+.4f}")
    
    print("\n" + "-" * 70)
    print("SANITY CHECK GUIDANCE")
    print("-" * 70)
    print("Expected patterns for creditworthiness:")
    print("  ✓ Higher income → POSITIVE coefficient")
    print("  ✓ Higher debt → NEGATIVE coefficient")
    print("  ✓ Debt-to-income ratio → NEGATIVE coefficient")
    print("  ✓ Payment history → POSITIVE coefficient")
    print("\nReview coefficients above to verify logic aligns with domain knowledge")
    print("=" * 70)


def analyze_prediction_uncertainty(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    uncertainty_threshold = 0.1
    uncertain_mask = (y_pred_proba > 0.5 - uncertainty_threshold) & (y_pred_proba < 0.5 + uncertainty_threshold)
    
    num_uncertain = uncertain_mask.sum()
    pct_uncertain = (num_uncertain / len(y_test)) * 100
    
    low_confidence = (y_pred_proba < 0.3) | (y_pred_proba > 0.7)
    medium_confidence = (y_pred_proba >= 0.3) & (y_pred_proba < 0.4) | (y_pred_proba > 0.6) & (y_pred_proba <= 0.7)
    high_confidence = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)
    
    print("\n" + "=" * 70)
    print("PREDICTION PROBABILITY ANALYSIS")
    print("=" * 70)
    
    print("\nCONFIDENCE DISTRIBUTION")
    print("-" * 70)
    print(f"Very Confident (< 0.3 or > 0.7):   {(~low_confidence).sum():5d} ({((~low_confidence).sum()/len(y_test)*100):.1f}%)")
    print(f"Moderately Confident (0.3-0.4, 0.6-0.7): {medium_confidence.sum():5d} ({(medium_confidence.sum()/len(y_test)*100):.1f}%)")
    print(f"Uncertain (0.4-0.6):               {high_confidence.sum():5d} ({(high_confidence.sum()/len(y_test)*100):.1f}%)")
    
    print("\nUNCERTAINTY ZONE (0.4 - 0.6)")
    print("-" * 70)
    print(f"Cases in uncertainty zone: {num_uncertain} ({pct_uncertain:.2f}%)")
    
    if pct_uncertain > 20:
        print("⚠️  HIGH UNCERTAINTY: >20% of predictions near decision boundary")
        print("   Consider: Feature engineering or ensemble methods")
    elif pct_uncertain > 10:
        print("⚠️  MODERATE UNCERTAINTY: Model struggles with some cases")
    else:
        print("✓  LOW UNCERTAINTY: Model makes confident predictions")
    
    print("\nPROBABILITY STATISTICS")
    print("-" * 70)
    print(f"Mean probability:   {y_pred_proba.mean():.4f}")
    print(f"Median probability: {np.median(y_pred_proba):.4f}")
    print(f"Std deviation:      {y_pred_proba.std():.4f}")
    print(f"Min probability:    {y_pred_proba.min():.4f}")
    print(f"Max probability:    {y_pred_proba.max():.4f}")
    print("=" * 70)
    
    return y_pred_proba


def threshold_sensitivity_analysis(y_test, y_pred_proba):
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    print("\n" + "=" * 70)
    print("DECISION THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nThreshold  Precision  Recall  F1-Score  TP    FP    FN    TN")
    print("-" * 70)
    
    results = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred_threshold, zero_division=0)
        recall = recall_score(y_test, y_pred_threshold, zero_division=0)
        f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred_threshold)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        marker = " ← DEFAULT" if threshold == 0.5 else ""
        print(f"{threshold:.2f}       {precision:.4f}     {recall:.4f}  {f1:.4f}    {tp:4d}  {fp:4d}  {fn:4d}  {tn:4d}{marker}")
    
    print("\n" + "-" * 70)
    print("THRESHOLD DECISION GUIDANCE")
    print("-" * 70)
    print("Lower threshold (0.3-0.4):")
    print("  → Higher Recall (catch more bad borrowers)")
    print("  → Lower Precision (more false alarms)")
    print("  → Use when: Cost of defaults > cost of rejecting good customers")
    print("\nHigher threshold (0.6-0.7):")
    print("  → Higher Precision (fewer false alarms)")
    print("  → Lower Recall (miss more bad borrowers)")
    print("  → Use when: Customer acquisition is critical")
    print("\nRecommendation: Choose threshold based on business cost ratio")
    print("  Cost_of_default / Cost_of_lost_customer = optimal_threshold_guide")
    print("=" * 70)
    
    return pd.DataFrame(results)


def main():
    print("\nLoading data and training baseline model...")
    X_train, X_test, y_train, y_test = load_train_test_data()
    
    model = train_baseline_model(X_train, y_train)
    
    feature_names = X_train.columns.tolist()
    
    feature_importance = extract_feature_importance(model, feature_names)
    
    analyze_coefficient_logic(feature_importance)
    
    y_pred_proba = analyze_prediction_uncertainty(model, X_test, y_test)
    
    threshold_results = threshold_sensitivity_analysis(y_test, y_pred_proba)
    
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE - READY FOR DAY 3 DECISION")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Verify coefficients align with credit risk domain knowledge")
    print("2. Assess uncertainty level - high uncertainty suggests complex patterns")
    print("3. Choose optimal threshold based on business constraints")
    print("4. If baseline ROC-AUC > 0.70, proceed to Random Forest")
    print("=" * 70 + "\n")
    
    return model, feature_importance, threshold_results


if __name__ == "__main__":
    trained_model, feature_analysis, threshold_analysis = main()
