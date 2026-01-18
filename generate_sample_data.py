import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

age = np.random.randint(18, 70, n_samples)
annual_income = np.random.lognormal(10.5, 0.5, n_samples)
employment_length = np.random.randint(0, 40, n_samples).astype(float)
total_debt = np.random.lognormal(9, 1.2, n_samples)
credit_history_length = np.random.randint(0, 30, n_samples)
num_credit_accounts = np.random.randint(1, 15, n_samples)
num_late_payments = np.random.poisson(2, n_samples)
credit_utilization = np.random.uniform(0, 1, n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
home_ownership = np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples, p=[0.3, 0.2, 0.5])

debt_to_income = total_debt / annual_income
risk_score = (
    -0.3 * (debt_to_income) +
    0.2 * (annual_income / 100000) +
    0.15 * (credit_history_length / 30) +
    -0.25 * (num_late_payments / 10) +
    -0.2 * (credit_utilization) +
    0.1 * (employment_length / 40) +
    np.random.normal(0, 0.15, n_samples)
)

creditworthy = (risk_score > 0).astype(int)

missing_mask = np.random.rand(n_samples, 3) < 0.05
annual_income[missing_mask[:, 0]] = np.nan
total_debt[missing_mask[:, 1]] = np.nan
employment_length[missing_mask[:, 2]] = np.nan

credit_data = pd.DataFrame({
    'age': age,
    'annual_income': annual_income,
    'employment_length': employment_length,
    'total_debt': total_debt,
    'credit_history_length': credit_history_length,
    'num_credit_accounts': num_credit_accounts,
    'num_late_payments': num_late_payments,
    'credit_utilization': credit_utilization,
    'education': education,
    'home_ownership': home_ownership,
    'creditworthy': creditworthy
})

credit_data.to_csv('credit_data.csv', index=False)

print(f"Generated credit_data.csv with {n_samples} samples")
print(f"\nClass distribution:")
print(credit_data['creditworthy'].value_counts())
print(f"\nMissing values:")
print(credit_data.isnull().sum())
