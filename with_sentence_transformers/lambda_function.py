import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model and label mappings
with open('../document_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../label_mappings.pkl', 'rb') as f:
    label_to_index, index_to_label = pickle.load(f)

# Load the Sentence Transformer model for encoding
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
        embedding = encoder_model.encode([document_text])

        # Make predictions
        prediction = model.predict(embedding)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
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
