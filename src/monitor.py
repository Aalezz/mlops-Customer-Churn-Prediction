import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, precision_score, recall_score
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

def load_test_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def evaluate_model_performance(model, X, y):
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        pass

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1_score": f1_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }
    if probs is not None:
        metrics["roc_auc"] = roc_auc_score(y, probs)
        metrics["log_loss"] = log_loss(y, probs)
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None
    return metrics

def main():
    mlflow.set_tracking_uri("C:\Users\HUAWEI\Desktop\Project-Al-ezz Al-dumaini-2101370\Project-Al-ezz Al-dumaini-2101370\my project\src\monitor.py")
    mlflow.set_experiment("Customer_Churn_Monitoring")

    # Replace this with the actual run_id of your best model from MLflow UI (from train.py or tune.py)
    run_id = "8f3b268e56824a5d9e61db94bc5dc175"

    # Load the model directly from the run's artifacts
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from run ID: {run_id}")
    model = mlflow.sklearn.load_model(model_uri)

    data_path = r"C:\Users\HUAWEI\Desktop\Project-Al-ezz Al-dumaini-2101370\Project-Al-ezz Al-dumaini-2101370\my project\data\WA_Fn-UseC_-Telco-Customer-Churn-processed.csv"
    X, y = load_test_data(data_path)

    metrics = evaluate_model_performance(model, X, y)
    print("Monitoring Metrics:", metrics)

    with mlflow.start_run(run_name="production_monitoring"):
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

if __name__ == "__main__":
    main()
