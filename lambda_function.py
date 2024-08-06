import json
import pickle
import boto3
import lightgbm as lgb
import numpy as np

# Load the model and label mappings
with open('document_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("Model loaded successfully.")

with open('label_mappings.pkl', 'rb') as f:
    label_to_index, index_to_label = pickle.load(f)
    print("Label mappings loaded successfully.")
    print(f"Label to index mapping: {label_to_index}")
    print(f"Index to label mapping: {index_to_label}")

def get_bedrock_embedding_chunked(text, max_chunk_length=8000, model_id="amazon.titan-embed-text-v2:0"):
    bedrock = boto3.client(service_name='bedrock-runtime')

    # Split the text into chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    print(f"Text split into {len(chunks)} chunks.")

    embeddings = []
    for i, chunk in enumerate(chunks):
        body = json.dumps({
            "inputText": chunk
        })
        print(f"Chunk {i+1} length: {len(chunk)} characters.")

        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response['body'].read())
        embeddings.append(response_body['embedding'])
        print(f"Received embedding for chunk {i+1}: {len(response_body['embedding'])} dimensions.")

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Average the embeddings if there are multiple chunks
    if embeddings_array.shape[0] > 1:
        averaged_embedding = np.mean(embeddings_array, axis=0)
        print(f"Averaged embedding shape: {averaged_embedding.shape}")
    else:
        averaged_embedding = embeddings_array[0]
        print(f"Single chunk embedding shape: {averaged_embedding.shape}")

    return averaged_embedding.tolist()

def lambda_handler(event, context):
    try:
        # Parse the request body
        print('Event:', event)
        body = json.loads(event['body'])
        print('Body:', body)
        print("Request body parsed successfully.")

        if 'document_text' not in body:
            print("Error: 'document_text' not found in request body.")
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Missing 'document_text' in request body"})
            }

        document_text = body['document_text']
        print(f"Document text length: {len(document_text)} characters.")
        print(f"Document text: {document_text[:100]}...")  # Print the first 100 characters for context

        # Encode the incoming document text
        embedding = get_bedrock_embedding_chunked(document_text)
        print(f"Embedding length: {len(embedding)} dimensions.")
        print(f"Embedding: {embedding[:10]}...")  # Print the first 10 dimensions for context

        # Make predictions
        prediction = model.predict([embedding])
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction: {prediction}")

        predicted_class_index = np.argmax(prediction[0])
        predicted_label = index_to_label[predicted_class_index]
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted label: {predicted_label}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": "Classification successful",
                "label": predicted_label
            })
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
