# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

st.title("Adult Income Prediction App")
st.write("Upload your test CSV and select a model to get predictions.")

# 1️⃣ Upload test dataset
uploaded_file = st.file_uploader("Upload CSV file with same columns as Adult dataset", type=["csv"])

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded!")
    st.write(f"Dataset shape: {df_test.shape}")

    # 2️⃣ Select model
    model_name = st.selectbox(
        "Select a trained model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # 3️⃣ Load the pre-trained model
    model_file = f"model/{model_name.replace(' ', '_')}.pkl"
    try:
        model = joblib.load(model_file)
        st.success(f"{model_name} loaded successfully!")
    except:
        st.error(f"Cannot load {model_file}. Make sure the model is trained and saved.")
        st.stop()

    # 4️⃣ Preprocess test data (label encoding for categorical columns)
    from model.models import preprocess_adult_data
    X_test, _ = preprocess_adult_data(df_test)  # ignore y

    # 5️⃣ Make predictions
    predictions = model.predict(X_test)
    df_test["Predicted_Income"] = ["<=50K" if x==0 else ">50K" for x in predictions]

    # 6️⃣ Display predictions
    st.subheader("Predictions")
    st.dataframe(df_test)
    st.success("Predictions completed!")
