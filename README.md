# Credit Scoring Model Project

A machine learning journey to predict whether someone is creditworthy—basically helping a bank decide who to trust with a loan.

## Project Goal

The core idea is simple: **can we predict if someone will repay a loan?** This isn't just theoretical—banks deal with this every day. They need to decide quickly whether to approve a loan or reject it. That's where machine learning comes in.

I built this project to automate that decision using historical data about past borrowers. By learning from patterns in the data (income, employment history, debt levels, payment history), the model can predict the creditworthiness of new applicants. It's not perfect, but it's way better than guessing.

## My Learning Journey (The 'Why')

### Data Cleaning: The Foundation

I started by realizing something crucial: **garbage in, garbage out**. If the data is messy, the model will be confused. So I spent time:

-   Identifying missing values (some people didn't fill out all fields)
-   Handling them carefully (using median for numbers, most common value for categories)
-   Encoding categorical features (converting text like "owns" vs "rents" into numbers the model could understand)

### The Data Leakage Problem (This Was Tricky)

This was the moment I really understood why proper data handling matters. **Data leakage** happens when you accidentally use information from your test set during training. For example, if I scaled the entire dataset before splitting it into train/test, the test data would influence the scaling parameters. That's cheating—it makes the model look better than it actually is.

So here's what I did right:

1. Split the data into train and test sets FIRST
2. Then fitted the scaler only on training data
3. Applied that scaler to test data

This way, the model truly sees test data for the first time during evaluation.

### The Debt-to-Income Ratio Insight

Early on, I had raw features like "total debt" and "annual income" separately. But I realized something: **context matters**. Someone with $50,000 debt making $200,000 a year is in a very different position than someone with $50,000 debt making $40,000 a year.

So I created a new feature: **Debt-to-Income Ratio** = Total Debt / Annual Income. This ratio captures relative financial burden in a way raw numbers don't. It turned out to be the single most important feature in my final model! This taught me that feature engineering—creating smarter features—is sometimes more powerful than adding more raw data.

## The Models I Tried

### Logistic Regression: The Baseline

I started with Logistic Regression because it's the "simple way"—it draws a straight line to separate creditworthy from non-creditworthy applicants. It's interpretable and fast.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Result:** ROC-AUC of **0.8473** (about 84.73%)

This became my baseline—the model I compared everything else against. It worked reasonably well, but I wondered if I could do better.

### Random Forest: The Smarter Team

Then I tried Random Forest. Instead of drawing one line, it builds 100 decision trees and lets them "vote" on the answer. Some trees focus on one pattern, others catch different patterns. The beauty is that they collectively make better decisions than any single tree.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

**Result:** ROC-AUC of **0.8119** (about 81.19%)

Interesting—it was actually slightly _lower_ than Logistic Regression out of the box. This taught me an important lesson: **more complex doesn't always mean better**. But then I tried hyperparameter optimization...

### Hyperparameter Optimization: Fine-Tuning the Model

I used RandomizedSearchCV to test 50 different combinations of parameters:

-   `n_estimators`: How many trees?
-   `max_depth`: How deep can each tree grow?
-   `min_samples_split`: When should a tree stop splitting?
-   `max_features`: Which features should each tree consider?

After testing 250 different configurations (50 combinations × 5-fold cross-validation), I found the best parameters.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

**Best Parameters Found:**

-   `n_estimators`: 250 (more trees = more perspectives)
-   `max_depth`: 5 (shallow trees prevent overfitting)
-   `min_samples_split`: 5 (require minimum samples before splitting)
-   `max_features`: sqrt (randomly consider square root of features)

**Result:** ROC-AUC of **0.8227** (about 82.27%)

Still not beating Logistic Regression, but getting closer. I realized that Random Forest might need more tuning, or that the dataset itself favors simpler models.

## Key Results

### What is ROC-AUC and Why Does It Matter?

Think of it like this: if you show the model 100 random pairs (one creditworthy person, one non-creditworthy person), how often does it correctly identify which is which? That percentage is what ROC-AUC measures. It ranges from 0 (always wrong) to 1.0 (always right).

**My Results:**

-   Logistic Regression: **0.8473** ✓ (Best)
-   Optimized Random Forest: **0.8227** (Close second)
-   Basic Random Forest: **0.8119** (Baseline Random Forest)

An ROC-AUC of 0.85 means the model is doing pretty well—better than random guessing (0.5) and in the "good" range for finance applications.

### The Confusion Matrix

At a 50% probability threshold, my optimized model showed:

```
                Predicted
                0    1
Actual    0   89   25  (True Negatives, False Positives)
          1   26   60  (False Negatives, True Positives)
```

This means:

-   **89 cases**: Correctly identified people who wouldn't default ✓
-   **60 cases**: Correctly identified people who would default ✓
-   **25 cases**: False alarms—approved people who ended up defaulting (BAD)
-   **26 cases**: Missed opportunities—rejected people who would have been fine (LOST MONEY)

## Risk Trade-offs: The Threshold Analysis

Here's where it got really interesting. I discovered that the model outputs a **probability** (0 to 1), and we get to choose where to draw the line.

### What I Discovered

If I set the threshold at **0.30** (very lenient):

-   Approve 75% of applicants
-   Bad loan risk: 57% (almost half the people we approve will default!)
-   But we don't miss good customers

If I set it at **0.50** (balanced):

-   Approve 42.5% of applicants
-   Bad loan risk: 22% (much safer)
-   Miss 30% of people who would have been fine

If I set it at **0.70** (very conservative):

-   Approve only 6% of applicants
-   Bad loan risk: Less than 1% (super safe)
-   But we reject 87% of people who would have been good customers

### The Real-World Impact

This was the moment I realized **there is no perfect answer**. A bank must choose:

| Threshold | Approval Rate | Default Risk | Missed Opportunity | Best For       |
| --------- | ------------- | ------------ | ------------------ | -------------- |
| 0.30      | 75%           | 57%          | 1%                 | Growth-focused |
| 0.50      | 42.5%         | 22%          | 30%                | Balanced       |
| 0.70      | 6%            | <1%          | 87%                | Risk-averse    |

A startup bank might choose 0.30 to grow fast. A conservative bank might choose 0.70 to protect capital. The model doesn't decide—it just shows the trade-offs.

## The Code Structure

I organized my work into modular scripts:

### 1. `data_preprocessing.py`

Handles data cleaning, imputation, encoding, and splitting.

### 2. `train_logistic_regression.py`

Trains and evaluates the baseline Logistic Regression model.

### 3. `train_tree_models.py`

Trains Decision Tree and Random Forest classifiers, extracts feature importances, and compares performance.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

print(f"Decision Tree ROC-AUC: {roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]):.4f}")
print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]):.4f}")
```

### 4. `optimize_random_forest.py`

Performs hyperparameter tuning and threshold analysis.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

for threshold in thresholds:
    y_pred_custom = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    print(f"Threshold {threshold}: Precision={precision:.4f}, Recall={recall:.4f}")
```

## Setup & Running the Code

### Prerequisites

```bash
python --version  # Requires Python 3.8+
```

### Installation

```bash
pip install pandas scikit-learn numpy
```

### Running the Scripts

```bash
python data_preprocessing.py           # Clean and prepare data
python train_logistic_regression.py    # Train baseline model
python train_tree_models.py            # Train tree-based models
python optimize_random_forest.py       # Hyperparameter tuning & threshold analysis
```

### Expected Output

Each script will display:

-   Model training progress
-   Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
-   Confusion matrices
-   Feature importances (for tree models)
-   Threshold analysis and bank risk assessment

## What I Learned

This project taught me way more than just "how to train a model." Here are the big takeaways:

1. **Data quality is everything** — Cleaning and engineering features took 60% of my time, but it was worth it
2. **Simple models can be powerful** — Logistic Regression beat my fancy Random Forest
3. **Context matters** — Creating the debt-to-income ratio was more impactful than using raw features
4. **There are always trade-offs** — No model is universally "right"; it depends on business goals
5. **Interpretability matters in finance** — A model that works isn't enough; stakeholders need to understand why

## Future Ideas

If I were to expand this project:

-   Add more features (credit card utilization patterns, loan history trends)
-   Try gradient boosting models (XGBoost, LightGBM)
-   Implement SHAP values to explain individual predictions
-   A/B test different thresholds in production
-   Track model performance over time (does it drift?)

## Repository

This project is hosted on GitHub: [CodeAlpha_Credit-Scoring-Model](https://github.com/ADAMSmugwe/CodeAlpha_Credit-Scoring-Model)

---

**A note to anyone reading this:** This is a learning project. It's meant to show my understanding of ML concepts, not to be production-grade code. In real banking, there are regulatory requirements, more data, and way more rigorous validation. But I'm proud of this work—it represents genuine learning, not just following tutorials.
