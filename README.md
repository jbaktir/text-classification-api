# Document Classification Lambda Function

This project implements a document classification system using AWS Lambda and Amazon Bedrock. The Lambda function is containerized and can be deployed to AWS Lambda.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── lambda_handler.py
├── embedding_utils.py
├── train_model.py
├── test_api_gateway.py
├── update_lambda.py
├── build_and_push_image.py
├── test_lambda_local.py
├── example_events.py
├── document_classification_model.pkl
└── label_mappings.pkl
```

## Setup and Deployment

1. Build and push the Docker image to ECR:
   ```
   python build_and_push_image.py
   ```

2. Update the Lambda function with the new image:
   ```
   python update_lambda.py
   ```

3. Set up an API Gateway trigger for the Lambda function if not already configured.

## Usage

Send a POST request to the API Gateway endpoint with the following JSON body:

```json
{
    "document_text": "Your document text here"
}
```

The response will contain the predicted label for the document.

## Training

To retrain the model:

1. Prepare your training data.
2. Update the `train_model.py` script as needed.
3. Run the training script:
   ```
   python train_model.py
   ```
4. Rebuild and redeploy the Docker image using the steps in the Setup and Deployment section.

## Testing

1. To test the Lambda function locally:
   ```
   python test_lambda_local.py
   ```

2. To test the deployed API Gateway endpoint:
   ```
   python test_api_gateway.py
   ```

## Updating the Lambda Function

If you need to update the Lambda function code:

1. Make your changes to the relevant Python files.
2. Rebuild and push the Docker image using `build_and_push_image.py`.
3. Update the Lambda function with the new image using `update_lambda.py`.

