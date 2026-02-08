# train_and_save_models.py

import pandas as pd
from model.models import preprocess_adult_data, get_models
import joblib
from sklearn.model_selection import train_test_split

# Load Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Preprocess
X, y = preprocess_adult_data(df)

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get models
models_dict = get_models()

# Train and save each model
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name.replace(' ', '_')}.pkl")
    print(f"{name} trained and saved!")
