# model/models.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_adult_data(df):
    df_clean = df.dropna()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')

    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

    if 'income' in df_clean.columns:
        df_clean['income'] = df_clean['income'].apply(lambda x: 0 if x == "<=50K" else 1)

    X = df_clean.drop(columns=['income'], errors='ignore')
    y = df_clean['income'] if 'income' in df_clean.columns else None
    return X, y

# -----------------------------
# Get Models
# -----------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

# -----------------------------
# Train and Evaluate
# -----------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred),4),
        "AUC": round(roc_auc_score(y_test, y_proba),4),
        "Precision": round(precision_score(y_test, y_pred),4),
        "Recall": round(recall_score(y_test, y_pred),4),
        "F1 Score": round(f1_score(y_test, y_pred),4),
        "MCC": round(matthews_corrcoef(y_test, y_pred),4)
    }

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, cr
