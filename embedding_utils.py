import boto3
import json
import numpy as np

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
        return np.mean(embeddings, axis=0).tolist()
    else:
        return embeddings[0]