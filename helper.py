import os
import glob
import pickle
import boto3
import json


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


def process_and_save_embeddings(parent_directory_path):
    # Gather all .txt files in the directory
    file_paths = glob.glob(os.path.join(parent_directory_path, '**/*.txt'), recursive=True)

    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_name = os.path.basename(file_path)
            subdirectory = os.path.basename(os.path.dirname(file_path))
            content = file.read().strip()
            data.append({'subdirectory': subdirectory, 'file_name': file_name, 'content': content})

    # Extract features and labels, excluding the 'other' category
    filtered_data = [item for item in data if item['subdirectory'] != 'other']
    contents = [item['content'] for item in filtered_data]
    labels = [item['subdirectory'] for item in filtered_data]
    embeddings = [get_bedrock_embedding_chunked(content) for content in contents]

    embedding_labels = list(zip(embeddings, labels))
    with open('embedding_labels.pkl', 'wb') as f:
        pickle.dump(embedding_labels, f)


def load_or_create_embeddings(parent_directory_path):
    if os.path.exists('embedding_labels.pkl'):
        with open('embedding_labels.pkl', 'rb') as f:
            embedding_labels = pickle.load(f)
    else:
        process_and_save_embeddings(parent_directory_path)
        with open('embedding_labels.pkl', 'rb') as f:
            embedding_labels = pickle.load(f)

    return embedding_labels
