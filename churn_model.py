import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'churn_model.pkl')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()

    # Drop customerID (not a feature)
    df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges (has spaces as missing values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # Label-encode remaining categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df


def train(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 1),
        'auc_roc':   round(roc_auc_score(y_test, y_prob) * 100, 1),
        'precision': round(precision_score(y_test, y_pred) * 100, 1),
        'recall':    round(recall_score(y_test, y_pred) * 100, 1),
        'f1':        round(f1_score(y_test, y_pred) * 100, 1),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])

    return metrics, cm, report


def feature_importance(model, X_train):
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    return joblib.load(path)


if __name__ == '__main__':
    print("Loading data...")
    df_raw = load_data()
    print(f"Dataset shape: {df_raw.shape}")

    print("Preprocessing...")
    df_clean = preprocess(df_raw)

    print("Training Random Forest...")
    model, X_train, X_test, y_train, y_test = train(df_clean)

    print("Evaluating...")
    metrics, cm, report = evaluate(model, X_test, y_test)

    print("\n--- Model Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}%")

    print("\n--- Confusion Matrix ---")
    print(cm)

    print("\n--- Classification Report ---")
    print(report)

    print("\n--- Top 10 Feature Importances ---")
    fi = feature_importance(model, X_train)
    print(fi.head(10).to_string(index=False))

    save_model(model)
