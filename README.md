# bits-ml-assignment-2
# Machine Learning Model Comparison

## a. Problem Statement
The objective of this assignment is to build and evaluate multiple machine learning classification models on a given dataset and compare their performance using standard evaluation metrics. The goal is to identify the most effective model based on predictive performance and robustness.

---

## b. Dataset Description
The dataset (Occupancy Detection) used in this project contains labeled data suitable for a supervised classification task.
It includes multiple input features (independent variables) collected from environmental sensors and one target variable (dependent variable) indicating room occupancy.

**Dataset Details:**
- Number of instances: 20560
- Number of features: 6
- Target variable: Occupancy (0:Not Occupied, 1: Occupied)
- Type of problem: Binary classification
- Source of dataset: UCI Machine Learning Repository

Basic preprocessing steps such as handling missing values, feature scaling, and data splitting were applied before model training.

---

## c. Models Used

The following machine learning models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Evaluation Metrics Used
The models were evaluated using the following metrics:
- Accuracy
- AUC (Area Under the ROC Curve)
- Precision
- Recall
- F1 Score
- MCC (Matthews Correlation Coefficient)

---

### Model Performance Comparison Table

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|---------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression |          |      |           |        |          |      |
| Decision Tree       |          |      |           |        |          |      |
| kNN                 |          |      |           |        |          |      |
| Naive Bayes         |          |      |           |        |          |      |
| Random Forest       |          |      |           |        |          |      |
| XGBoost             |          |      |           |        |          |      |

---

## Model Performance Observations

| ML Model Name        | Observation about Model Performance |
|---------------------|-------------------------------------|
| Logistic Regression |                                     |
| Decision Tree       |                                     |
| kNN                 |                                     |
| Naive Bayes         |                                     |
| Random Forest       |                                     |
| XGBoost             |                                     |

---

## Conclusion
Based on the evaluation metrics and observations, the ensemble models (Random Forest and XGBoost) generally perform better due to their ability to capture complex patterns and reduce overfitting. The final model selection depends on the balance between performance, interpretability, and computational efficiency.


