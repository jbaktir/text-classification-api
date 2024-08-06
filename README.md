Sure, I can update the README to explain the new files. Here is the revised version:

---

# AWS Lambda Function with API Gateway

This project demonstrates how to create and deploy an AWS Lambda function using Docker. The Lambda function is served via API Gateway. The function is designed for document classification, utilizing a pre-trained machine learning model.

## Prerequisites

- AWS CLI
- Docker
- AWS Account

## Project Structure

```
.
├── Dockerfile
├── lambda_function.py
├── document_classification_model.pkl
├── label_mappings.pkl
├── requirements.txt
├── README.md
├── update_lambda_function.py
├── test_lambda_function.py
├── test_api_gateway.py
├── example_events.py
├── helper.py
├── train.py
└── with_sentence_transformers
    └── train.py
```

## Dockerfile

The Dockerfile sets up the environment, installs dependencies, and specifies the handler for the Lambda function.

```Dockerfile
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ make

# Copy function code and model files
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY document_classification_model.pkl ${LAMBDA_TASK_ROOT}
COPY label_mappings.pkl ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
```

## Steps to Build and Deploy

### 1. Build the Docker Image

Navigate to the project directory and build the Docker image:

```sh
docker build -t my-lambda-image .
```

### 2. Test the Docker Image Locally (Optional)

Run the Docker container locally to test the Lambda function:

```sh
docker run -p 9000:8080 my-lambda-image
```

Invoke the function locally:

```sh
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'
```

### 3. Push the Docker Image to Amazon ECR

Create a repository in Amazon ECR and push the Docker image:

```sh
# Authenticate Docker to your ECR registry
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

# Tag the image
docker tag my-lambda-image:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest

# Push the image
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest
```

### 4. Create the Lambda Function

Create the Lambda function using the pushed Docker image:

```sh
aws lambda create-function \
    --function-name my-document-classifier \
    --package-type Image \
    --code ImageUri=<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest \
    --role arn:aws:iam::<your-account-id>:role/<your-lambda-execution-role>
```

### 5. Set Up API Gateway

Create an API Gateway and configure it to invoke your Lambda function. Detailed steps can be found in the [AWS documentation](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html).

## Lambda Function Handler

The Lambda function is defined in `lambda_function.py` and is set to use the `lambda_handler` function as the entry point.

```python
import json
import pickle

def lambda_handler(event, context):
    # Load the model and mappings
    with open('document_classification_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('label_mappings.pkl', 'rb') as mappings_file:
        label_mappings = pickle.load(mappings_file)

    # Process the input event
    document = event['document']
    prediction = model.predict([document])
    label = label_mappings[prediction[0]]
    
    return {
        'statusCode': 200,
        'body': json.dumps({'label': label})
    }
```

## Explanation of Additional Files

### update_lambda_function.py

This script automates the process of updating the Lambda function code. It packages the necessary files into a zip file and updates the Lambda function.

### test_lambda_function.py

This script is used for testing the Lambda function locally by invoking it with predefined events.

### test_api_gateway.py

This script tests the API Gateway by sending a request to the endpoint and printing the response.

### example_events.py

This file contains example events that can be used to test the Lambda function.

### helper.py

This script includes helper functions to process documents, generate embeddings, and save the processed data.

### train.py

This script trains the machine learning model using the processed data. It utilizes LightGBM and Optuna for hyperparameter optimization.

### with_sentence_transformers/train.py

This script provides an alternative training approach using Sentence Transformers to generate embeddings for the documents. It also trains a LightGBM model with the generated embeddings.

---
