import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
sonar_data = pd.read_csv("sonar_data.csv", header=None)

# Splitting features and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

print(f"✅ Accuracy on training data: {train_accuracy:.2f}")
print(f"✅ Accuracy on test data: {test_accuracy:.2f}")

# 🔹 Check predictions on first 5 test samples
print("Sample Predictions on Test Data:", model.predict(X_test[:5]))  

# Save model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and scaler saved as model.pkl and scaler.pkl")
