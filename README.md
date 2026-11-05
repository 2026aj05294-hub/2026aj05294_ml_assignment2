# Telco Customer Churn - ML Classification Deployment

### a. Problem Statement
The objective of this project is to build and evaluate multiple Machine Learning classification models to predict customer churn in a telecommunications company. By analyzing customer demographics, account information, and subscribed services, the models aim to identify customers who are highly likely to leave, allowing the business to take proactive retention measures. The final chosen model is deployed via a Streamlit web application.

### b. Dataset Description
- **Source:** Telco Customer Churn Dataset (Kaggle/IBM)
- **Context:** Predict behavior to retain customers.
- **Features:** Includes 21 columns (demographics like gender, SeniorCitizen, partner status, account details like tenure, contract type, payment method, monthly charges, and services signed up for).
- **Target Variable:** `Churn` (Yes = 1, No = 0). 
- **Instances:** 7,043 rows. Minimum thresholds (Features > 12, Instances > 500) are met.

### c.a Github Repository Link
*https://github.com/2026aj05294-hub/2026aj05294_ml_assignment2/tree/main*

### c.b Streamlit App Link
*https://2026aj05294mlassignment2-uhn6z6y3aidppbgwrzf7er.streamlit.app*

### d. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8197 | 0.8620 | 0.6831 | 0.5952 | 0.6361 | 0.5192 |
| **Decision Tree** | 0.7083 | 0.6283 | 0.4495 | 0.4531 | 0.4513 | 0.2526 |
| **KNN** | 0.7708 | 0.7900 | 0.5740 | 0.5201 | 0.5457 | 0.3938 |
| **Naive Bayes** | 0.6657 | 0.8377 | 0.4359 | 0.8928 | 0.5858 | 0.4222 |
| **Support Vector Machine** | 0.8141 | 0.8230 | 0.6934 | 0.5335 | 0.6030 | 0.4914 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed the best overall with the highest Accuracy (81.9%) and AUC (86.2%). It balances precision and recall well for this linear dataset. |
| **Decision Tree** | Showed the poorest performance (Accuracy 70.8%, MCC 0.25), likely suffering from overfitting on the training data and failing to generalize. |
| **KNN** | Moderate performance but struggled with precision (57.4%). Distance-based metrics were likely affected by the high dimensionality of the one-hot encoded categorical data. |
| **Naive Bayes** | Had the lowest accuracy (66.5%) but an exceptionally high recall (89.2%). It is overly eager to predict churn, resulting in many false positives. |
| **Support Vector Machine** | Strong performance nearly matching Logistic Regression (Accuracy 81.4%, AUC 82.3%), handling the non-linear boundaries well but taking longer to train. |
| **Overall Winner** | **Logistic Regression**. It provides the highest accuracy, the highest AUC score, and the best MCC, making it the most robust and reliable model for this dataset. |
