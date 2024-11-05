
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

data = pd.read_csv("C:\\Users\\jithy\Desktop\\customer_churn_prediction\customer_churn (1).csv")


print(data.head())


if 'signup_date' in data.columns:
    data['signup_date'] = pd.to_datetime(data['signup_date'])  # Convert to datetime
    # Extract useful features from the date
    data['signup_year'] = data['signup_date'].dt.year
    data['signup_month'] = data['signup_date'].dt.month
    data['signup_day'] = data['signup_date'].dt.day
    data['signup_hour'] = data['signup_date'].dt.hour
    data['signup_dayofweek'] = data['signup_date'].dt.dayofweek
    # Drop the original date column if it's no longer needed
    data.drop('signup_date', axis=1, inplace=True)


data = pd.get_dummies(data) 

X = data.drop('Age', axis=1)  
y = data['Age']               


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(model, 'customer_churn_model.pkl')

print("Model trained and saved as 'customer_churn_model.pkl'.")

