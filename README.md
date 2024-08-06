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

## Data Preparation and Embedding

Before training the model, you need to prepare your data and generate embeddings. The project assumes that your training data is organized in subdirectories, where each subdirectory name represents a label category.

1. Organize your text documents in subdirectories, with each subdirectory representing a category.

2. Use the `process_and_save_embeddings` function in `embedding_utils.py` to generate embeddings:

   ```python
   from embedding_utils import process_and_save_embeddings

   process_and_save_embeddings('path/to/your/data/directory')
   ```

   This function will:
   - Recursively find all .txt files in the specified directory
   - Generate embeddings for each document using Amazon Bedrock
   - Save the embeddings and corresponding labels to 'embedding_labels.pkl'

## Training

To train the model:

1. Ensure you have generated the embeddings as described in the Data Preparation and Embedding section.

2. Run the training script:
   ```
   python train_model.py
   ```

The training process uses Optuna for hyperparameter optimization and trains a LightGBM model. The script will:
- Load the embeddings and labels from 'embedding_labels.pkl'
- Split the data into training and testing sets
- Use Optuna to find the best hyperparameters
- Train the final model with the best hyperparameters
- Evaluate the model's accuracy
- Save the trained model as 'document_classification_model.pkl'
- Save the label mappings as 'label_mappings.pkl'

After training, you should rebuild and redeploy the Docker image to update the Lambda function with the new model.

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

## Note

Make sure to set up the necessary AWS credentials and permissions before running the scripts. You may need to configure the AWS CLI or set environment variables for AWS access. Ensure that you have the required permissions to use Amazon Bedrock for generating embeddings.