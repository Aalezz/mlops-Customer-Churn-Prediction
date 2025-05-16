import requests
import json

url = "http://127.0.0.1:1234/invocations"  

with open("payload.json") as f:
    data = json.load(f)

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, json=data)

print("Response status:", response.status_code)
print("Prediction response:", response.json())