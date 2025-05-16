import mlflow
import mlflow.sklearn
import pandas as pd
from data_processing import load_data
from sklearn.model_selection import train_test_split

model_uri = "runs:/8379585a9de945c28854a828bc0938f8/model" 

# Load and split the dataset
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pick one random row from the test set
sample = X_test.sample(n=1, random_state=42)
print("Sample input:")
print(sample)

# Load the model
model = mlflow.sklearn.load_model(model_uri)

# Predict
prediction = model.predict(sample)
print("Prediction:", prediction)
