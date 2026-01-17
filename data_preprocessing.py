import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath):
    return pd.read_csv(filepath)


def identify_column_types(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_columns, categorical_columns


def impute_missing_values(df, numerical_columns, categorical_columns):
    df_clean = df.copy()
    
    if numerical_columns:
        numerical_imputer = SimpleImputer(strategy='median')
        df_clean[numerical_columns] = numerical_imputer.fit_transform(df_clean[numerical_columns])
    
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_columns] = categorical_imputer.fit_transform(df_clean[categorical_columns])
    
    return df_clean


def encode_categorical_features(df, categorical_columns):
    df_encoded = df.copy()
    
    binary_columns = [col for col in categorical_columns 
                      if df[col].nunique() == 2]
    
    multi_category_columns = [col for col in categorical_columns 
                             if df[col].nunique() > 2]
    
    for col in binary_columns:
        label_encoder = LabelEncoder()
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    
    if multi_category_columns:
        df_encoded = pd.get_dummies(df_encoded, columns=multi_category_columns, drop_first=True)
    
    return df_encoded


def create_debt_to_income_ratio(df, debt_column='total_debt', income_column='annual_income'):
    df_featured = df.copy()
    df_featured['debt_to_income_ratio'] = df_featured[debt_column] / df_featured[income_column]
    df_featured['debt_to_income_ratio'] = df_featured['debt_to_income_ratio'].replace([np.inf, -np.inf], np.nan)
    df_featured['debt_to_income_ratio'].fillna(df_featured['debt_to_income_ratio'].median(), inplace=True)
    return df_featured


def split_features_target(df, target_column='creditworthy'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def analyze_class_balance(y):
    value_counts = y.value_counts()
    proportions = y.value_counts(normalize=True)
    
    print("=" * 50)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 50)
    print(f"\nClass Distribution:\n{value_counts}")
    print(f"\nClass Proportions:\n{proportions}")
    
    minority_class_ratio = proportions.min()
    if minority_class_ratio < 0.3:
        print(f"\n⚠️  WARNING: Significant class imbalance detected!")
        print(f"Minority class ratio: {minority_class_ratio:.2%}")
        print("Consider using SMOTE, class weights, or resampling tomorrow.")
    else:
        print(f"\n✓ Classes are reasonably balanced ({minority_class_ratio:.2%})")
    
    return value_counts, proportions


def preprocess_credit_data(filepath, target_column='creditworthy', test_size=0.2, random_state=42):
    df = load_data(filepath)
    
    numerical_columns, categorical_columns = identify_column_types(df)
    
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    df_clean = impute_missing_values(df, numerical_columns, categorical_columns)
    
    df_encoded = encode_categorical_features(df_clean, categorical_columns)
    
    df_featured = create_debt_to_income_ratio(df_encoded)
    
    X, y = split_features_target(df_featured, target_column)
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_column)
        print(f"\nTarget encoded: {dict(enumerate(label_encoder.classes_))}")
    
    class_counts, class_props = analyze_class_balance(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_credit_data('credit_data.csv')
    
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nFeatures: {X_train.columns.tolist()}")
    print(f"\nMissing values in train: {X_train.isnull().sum().sum()}")
    print(f"Missing values in test: {X_test.isnull().sum().sum()}")
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('credit_data_train.csv', index=False)
    test_data.to_csv('credit_data_test.csv', index=False)
    
    print(f"\n✓ Saved: credit_data_train.csv")
    print(f"✓ Saved: credit_data_test.csv")
