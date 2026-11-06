import io

import pandas as pd
import joblib
import os

import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Create models directory
os.makedirs('model', exist_ok=True)

# 1. Load the data
response = requests.get('https://raw.githubusercontent.com/2026aj05294-hub/2026aj05294_ml_assignment2/refs/heads/main/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(io.StringIO(response.text))

# 2. Preprocessing
# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop('customerID', axis=1)  # Drop ID column

# Encode Target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# One-Hot Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed test data as a CSV for the Streamlit app
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['Churn'] = y_test.values
test_df.to_csv('test_data.csv', index=False)

# 5. Initialize Models
models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive_Bayes': GaussianNB(),
    'SVM': CalibratedClassifierCV(SVC(random_state=42), ensemble=True)
}

# 6. Train, Evaluate, and Save Models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Save the model
    joblib.dump(model, f'model/{name}.pkl')

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics calculation
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })
    print(f"Successfully trained and saved {name}")

# Print metrics
results_df = pd.DataFrame(results)
print("\nModel Evaluation Metrics:")
print(results_df.round(4).to_markdown(index=False))
