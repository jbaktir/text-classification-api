import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("API_GATEWAY_URL")
api_key = os.getenv("LAMBDA_API_KEY")

body = {
    "document_text": """Your test document text here"""
}

headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(body))

print(response.status_code)
print(response.json())