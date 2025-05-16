import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.models.signature as signature
import warnings

warnings.filterwarnings("ignore")


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        
        pass

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else None

    return acc, f1, auc


def main():
    mlflow.set_tracking_uri("file:///D:/AALEZZ MLOPS - Copy/mlruns")
    mlflow.set_experiment("Customer_Churn_MultiModel")

    data_path = r"D:\AALEZZ MLOPS - Copy\mlflow-churn-project-main\data\WA_Fn-UseC_-Telco-Customer-Churn-processed.csv"
    X, y = load_data(data_path)

    # Check data quality
    if X.isnull().any().any():
        print("Warning: Missing values detected in features. Please handle before training.")
    print("Feature types:")
    print(X.dtypes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)

                acc, f1, auc = evaluate_model(model, X_test, y_test)

                mlflow.log_param("model_name", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                if auc is not None:
                    mlflow.log_metric("roc_auc", auc)

                input_example = X_train.head(5)
                sig = signature.infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=sig)

                print(f"{name} --> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc if auc else 'N/A'}")

            except Exception as e:
                print(f"Error training {name}: {e}")


if __name__ == "__main__":
    main()