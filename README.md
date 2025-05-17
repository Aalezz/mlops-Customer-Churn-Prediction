# Customer Churn Prediction with MLflow Lifecycle Management

<div align="center">
  
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white)

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)  
- [Dataset](#-dataset)  
- [Environment Setup](#-environment-setup)  
- [Data Preprocessing](#-data-preprocessing)  
- [Model Training](#-model-training)  
- [Hyperparameter Tuning](#-hyperparameter-tuning)  
- [Model Registration and Deployment](#-model-registration-and-deployment)  
- [Model Serving and API Testing](#-model-serving-and-api-testing)  
- [Model Monitoring](#-model-monitoring)  
- [References](#-references)  

## 🔍 Project Overview

This project builds and manages customer churn prediction models using various machine learning algorithms including Logistic Regression, Random Forest, XGBoost, and SVM. MLflow is used to log experiments, register models, and serve predictions. Hyperopt automates hyperparameter tuning, improving model accuracy. A monitoring framework tracks the production model's health over time.

## 📊 Dataset

- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Description**: Contains customer demographics, account information, and service usage metrics for churn prediction.

## 🛠️ Environment Setup

- Python 3.10.x  
- Virtual environment created using `venv`  
- Key libraries: pandas, scikit-learn, xgboost, mlflow, hyperopt, requests  
- MLflow tracking URI set to local folder:  
  ```python
  mlflow.set_tracking_uri("file:///D:/AALEZZ MLOPS - Copy/mlflow-churn-project-main/mlruns")
  ```
- MLflow UI accessible at http://localhost:5000

## 🧹 Data Preprocessing

- Dropped irrelevant columns like customerID
- Converted categorical variables to numeric format using one-hot and label encoding
- Ensured dataset has no missing values and all features are numeric
- Saved the processed dataset for modeling

## 🧠 Model Training

- Trained multiple models: Logistic Regression, Random Forest, XGBoost, and SVM
- Evaluated using accuracy, F1-score, and ROC AUC
- Logged all runs, parameters, metrics, and models with MLflow
- Included model input examples and signatures for reproducibility

Run training:
```bash
python src/train.py
```

## ⚙️ Hyperparameter Tuning

- Used Hyperopt for automated tuning of Random Forest hyperparameters
- Each trial logged as a nested MLflow run
- Identified best hyperparameters improving ROC AUC
- Streamlined model optimization process

Run tuning:
```bash
python src/tune.py
```

## 📦 Model Registration and Deployment

- Registered the best model version in MLflow Model Registry as `Customer_Churn_BestModel`
- Promoted the best version to Production stage
- Enabled version control and stage transitions

Register and promote:
```bash
python register_and_promote.py
```

Serve the production model:
```bash
mlflow models serve -m "models:/Customer_Churn_BestModel/Production" -p 1234 --no-conda
```

## 🌐 Model Serving and API Testing

- Served model locally on port 1234 with MLflow
- Tested REST API using a JSON payload (`payload.json`) and Python script (`test-api.py`)
- Confirmed predictions via API response

Test API:
```bash
python test-api.py
```

## 📈 Model Monitoring

- Periodically evaluated production model on validation data
- Logged monitoring metrics in MLflow under a dedicated experiment
- Enabled tracking of model performance and early detection of drift

Run monitoring:
```bash
python src/monitor.py
```

## 📚 References

- Blastchar, "Telco Customer Churn," Kaggle, 2018.
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- Aalezz, "MLflow Churn Project," GitHub Repository, 2025.
  https://github.com/Aalezz/mlflow-churn-project

- MLflow Documentation: https://mlflow.org/docs/latest/index.html

- Hyperopt Documentation: http://hyperopt.github.io/hyperopt/
