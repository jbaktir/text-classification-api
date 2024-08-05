import json
import pickle
import boto3

import os
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.getcwd(), 'lib')

# Load the model and label mappings
with open('document_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_mappings.pkl', 'rb') as f:
    label_to_index, index_to_label = pickle.load(f)

def get_bedrock_embedding_chunked(text, max_chunk_length=8000, model_id="amazon.titan-embed-text-v2:0"):
    bedrock = boto3.client(service_name='bedrock-runtime')

    # Split the text into chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    embeddings = []
    for chunk in chunks:
        body = json.dumps({
            "inputText": chunk
        })

        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response['body'].read())
        embeddings.append(response_body['embedding'])

    # Average the embeddings if there are multiple chunks
    if len(embeddings) > 1:
        return average_embeddings(embeddings)
    else:
        return embeddings[0]

def average_embeddings(embeddings):
    num_embeddings = len(embeddings)
    embedding_length = len(embeddings[0])
    averaged_embedding = [0] * embedding_length

    for embedding in embeddings:
        for i in range(embedding_length):
            averaged_embedding[i] += embedding[i]

    for i in range(embedding_length):
        averaged_embedding[i] /= num_embeddings

    return averaged_embedding

def lambda_handler(event, context):
    try:
        # Parse the request body
        body = json.loads(event['body'])

        if 'document_text' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Missing 'document_text' in request body"})
            }

        document_text = body['document_text']

        # Encode the incoming document text
        embedding = get_bedrock_embedding_chunked(document_text)

        # Make predictions
        prediction = model.predict([embedding])
        predicted_class_index = max(range(len(prediction[0])), key=prediction[0].__getitem__)
        predicted_label = index_to_label[predicted_class_index]

        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": "Classification successful",
                "label": predicted_label
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
