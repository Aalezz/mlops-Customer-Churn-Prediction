import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = r"C:\Users\HUAWEI\Desktop\Project-Al-ezz Al-dumaini-2101370\Project-Al-ezz Al-dumaini-2101370\my project\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# 1. Drop customerID
df = df.drop(columns=['customerID'])

# Helper function to map Yes/No to 1/0, also handle No phone/internet service
def binary_map(val):
    if val in ['Yes', 'DSL', 'Fiber optic', 'Male', 'Month-to-month', 'Electronic check']:
        return 1
    else:
        return 0

# 2. gender: Male=1, Female=0
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# 3. Partner: Yes=1, No=0
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})

# 4. Dependents: Yes=1, No=0
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})

# 5. PhoneService: Yes=1, No=0
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})

# 6. MultipleLines: Yes=1, No phone service=1, No=0
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No phone service': 1, 'No': 0})

# 7. InternetService: DSL or Fiber optic=1, No=0
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 1, 'No': 0})

# For columns with "Yes", "No", or "No internet service" values, map as 1 or 0:
cols_with_no_internet_service = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

for col in cols_with_no_internet_service:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

# 14. PaperlessBilling: Yes=1, No=0
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# 15. Churn: Yes=1, No=0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Label encode Contract and PaymentMethod
label_enc_contract = LabelEncoder()
df['Contract'] = label_enc_contract.fit_transform(df['Contract'])

label_enc_payment = LabelEncoder()
df['PaymentMethod'] = label_enc_payment.fit_transform(df['PaymentMethod'])

# Ensure TotalCharges is numeric (some values might be blank or whitespace)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with 0 or mean as appropriate
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Check final dtypes
print(df.dtypes)

# Preview the processed dataframe
print(df.head())

# Save the processed DataFrame to a new CSV file
processed_file_path = r"C:\Users\HUAWEI\Desktop\Project-Al-ezz Al-dumaini-2101370\Project-Al-ezz Al-dumaini-2101370\my project\data\WA_Fn-UseC_-Telco-Customer-Churn-processed.csv"
df.to_csv(processed_file_path, index=False)

print(f"Processed dataset saved to: {processed_file_path}")
