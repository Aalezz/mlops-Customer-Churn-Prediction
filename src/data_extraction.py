import os
import kaggle

# Set Kaggle API key directory
os.environ['KAGGLE_CONFIG_DIR'] = r'C:\\Users\\LEGION\\.kaggle'

# Destination folder for data
download_path = 'data'
os.makedirs(download_path, exist_ok=True)

# Download and unzip the dataset from Kaggle
kaggle.api.dataset_download_files(
    'blastchar/telco-customer-churn',
    path=download_path,
    unzip=True
)

print("✅ Dataset downloaded and extracted to:", download_path)
