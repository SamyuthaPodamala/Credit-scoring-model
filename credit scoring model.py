import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url="/content/german.data.txt"
columns = ["checking_account_status", "duration", "credit_history", "purpose", "credit_amount",
           "savings_account", "employment_status", "installment_rate", "personal_status",
           "other_debtors", "residence_since", "property", "age", "other_installment_plans",
           "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker", "credit_risk"]
data = pd.read_csv(url, sep=' ', names=columns)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target variable
X = data.drop('credit_risk', axis=1)
y = data['credit_risk']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))