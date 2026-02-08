# sample_run_models.py
from sklearn.model_selection import train_test_split
import pandas as pd
from models import get_models, preprocess_adult_data, train_and_evaluate

# -----------------------------
# 1️ Load Adult dataset
# -----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Preprocess
X, y = preprocess_adult_data(df)

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mdls=get_models()
# -----------------------------
# 3️ Train, evaluate and store results
# -----------------------------
results = []

for name, model in mdls.items():
    metrics, _, _ = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    results.append({
        "ML Model Name": name,
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics["AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1 Score"],
        "MCC": metrics["MCC"]
    })

# -----------------------------
# 4️ Display comparison table
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df[
    ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]

print("Comparison of ML Models on Adult Dataset:\n")
print(results_df)
results_df.to_csv("./results.csv")