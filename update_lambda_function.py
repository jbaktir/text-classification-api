import boto3
import os
import zipfile


def update_lambda_function(function_name, file_paths):
    # Initialize a session using Amazon Lambda
    lambda_client = boto3.client('lambda')

    # Create a temporary zip file
    zip_file_path = 'lambda_function.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in file_paths:
            if os.path.isfile(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    # Read the zip file content
    with open(zip_file_path, 'rb') as zip_file:
        zipped_code = zip_file.read()

    # Update the Lambda function code
    response = lambda_client.update_function_code(
        FunctionName=function_name,
        ZipFile=zipped_code
    )

    # Remove the temporary zip file
    os.remove(zip_file_path)

    return response


function_name = 'classify_document'
file_paths = [
    'lambda_function.py',
    'document_classification_model.pkl',
    'label_mappings.pkl'
]
update_lambda_function(function_name, file_paths)
