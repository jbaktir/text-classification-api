import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("API_GATEWAY_URL")
api_key = os.getenv("LAMBDA_API_KEY")

if not url or not api_key:
    raise ValueError("API_GATEWAY_URL and LAMBDA_API_KEY must be set in the .env file")

print(f"URL: {url}")
print(f"API Key: {api_key[:5]}...")  # Print first 5 characters of API key for verification

body = {
    "document_text": """Replace this with your actual test document text"""
}

headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(body))
    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print("Response content:")
        print(e.response.text)
