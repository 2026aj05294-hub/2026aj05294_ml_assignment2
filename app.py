import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
                y_prob = [0] * len(y_test)

            # Calculate Metrics
            st.write(f"### Evaluation Metrics for {selected_model_name.replace('_', ' ')}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col2.metric("AUC Score", round(roc_auc_score(y_test, y_prob), 4))
            col3.metric("Precision", round(precision_score(y_test, y_pred), 4))

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
            col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
            col6.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))

            st.divider()

            col_cm, col_cr = st.columns(2)

            with col_cm:
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                # Create a heatmap using seaborn
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Retained (0)', 'Churned (1)'],
                            yticklabels=['Retained (0)', 'Churned (1)'])
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)

            with col_cr:
                st.write("### Classification Report")
                # Convert the classification report to a dictionary, then to a DataFrame
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()

                # Display as a styled dataframe with 4 decimal places
                report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        else:
            st.error("Model file not found. Ensure models are inside the 'model' directory.")
    else:
        st.error("The uploaded CSV must contain the 'Churn' target column.")
else:
    st.info("Please upload the test_data.csv file from the sidebar to proceed.")