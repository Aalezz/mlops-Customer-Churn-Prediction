import mlflow
from mlflow.tracking import MlflowClient
import time

# Assume you have an active MLflow run here from training
run_id = mlflow.active_run().info.run_id
model_name = "Customer_Churn_BestModel"
model_uri = f"D:\AALEZZ MLOPS - Copy\mlflow-churn-project-main\mlruns\200416519386102702\1cfbba176fb14420aeedde35053ba431\artifacts\model\MLmodel"

client = MlflowClient()

print(f"Registering model from run: {run_id}")
model_version = mlflow.register_model(model_uri, model_name)

time.sleep(10)  # Wait for registration to complete

print(f"Promoting model version {model_version.version} to Production")
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)

print("Done.")