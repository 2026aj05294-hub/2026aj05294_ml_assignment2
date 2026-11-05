import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix, classification_report

st.title("Telco Customer Churn Prediction App")
st.write("Upload your test data to evaluate the classification models.")

# Sidebar for file upload and model selection
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Available models
model_names = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Naive_Bayes', 'SVM']
selected_model_name = st.sidebar.selectbox("Select ML Model", model_names)

if uploaded_file is not None:
    # Load test data
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_data.head())

    # Separate Features and Target
    if 'Churn' in test_data.columns:
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']

        # Load the selected model
        model_path = os.path.join('model', f'{selected_model_name}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            # Make Predictions
            y_pred = model.predict(X_test)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_prob = [0] * len(y_test)  # Fallback if probability is not supported

            # Calculate Metrics
            st.write(f"### Evaluation Metrics for {selected_model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col2.metric("AUC Score", round(roc_auc_score(y_test, y_prob), 4))
            col3.metric("Precision", round(precision_score(y_test, y_pred), 4))

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
            col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
            col6.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))

            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.code(cm)

            # Classification Report
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

        else:
            st.error("Model file not found. Ensure models are inside the 'model' directory.")
    else:
        st.error("The uploaded CSV must contain the 'Churn' target column.")
else:
    st.info("Please upload the test_data.csv file from the sidebar to proceed.")