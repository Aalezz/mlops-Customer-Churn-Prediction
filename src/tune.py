import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.sklearn
import mlflow.models.signature as signature
import warnings

warnings.filterwarnings("ignore")

# Load dataset
data_path = r"D:\AALEZZ MLOPS - Copy\mlflow-churn-project-main\data\WA_Fn-UseC_-Telco-Customer-Churn-processed.csv"
df = pd.read_csv(data_path)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def objective(params):
    with mlflow.start_run(nested=True):
        # Convert hyperopt floats to int where necessary
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        input_example = X_train.head(5)
        sig = signature.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=sig)

        print(f"Trial with params: {params} -> AUC: {auc:.4f}")

        # We want to minimize loss for hyperopt, so use negative AUC
        return {'loss': -auc, 'status': STATUS_OK}


search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
}

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///D:/AALEZZ MLOPS - Copy/mlruns")
    mlflow.set_experiment("Customer_Churn_Hyperopt_Tuning")

    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30,   # Increase for better search, reduce for faster tuning
        trials=trials,
        rstate=None,
    )

    print("Best hyperparameters:", best)