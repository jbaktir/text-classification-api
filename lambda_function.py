import json
import pickle
import numpy as np
from embedding_utils import get_bedrock_embedding_chunked

# Load the model and label mappings
with open('document_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_mappings.pkl', 'rb') as f:
    label_to_index, index_to_label = pickle.load(f)

def classify_document(embedding, model, index_to_label):
    # Make prediction
    prediction = model.predict([embedding])
    predicted_class_index = np.argmax(prediction[0])
    predicted_label = index_to_label[predicted_class_index]
    return predicted_label

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

        # Get embedding for the document text
        embedding = get_bedrock_embedding_chunked(document_text)

        # Classify the document
        predicted_label = classify_document(embedding, model, index_to_label)

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