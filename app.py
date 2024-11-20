# imports 
from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer
import torch
import os
from datetime import datetime

# initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# initalize elasticsearch client
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
client = Elasticsearch(ES_HOST)

# test connection
try:
    if not client.ping():
        print("Elasticsearch is not available. Check your configuration.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")


# define mappings
mappings = {
    "properties": {
        "class": {
            "type": "text"
        },
        "date": {
            "type": "date"
        },
        "topic_vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": "true",
            "similarity": "cosine",
        },
        "topic": {
            "type": "text"
        },
        "notes_vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": "true",
            "similarity": "cosine",
        },
        "notes": {
            "type": "text"
        }
    }
}

index_settings = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    },
    "mappings": mappings
}

# create index if needed 
RESET_INDEX = os.getenv("RESET_INDEX", "false").lower() == "true"
if RESET_INDEX:
    client.indices.delete(index="notebook_index", ignore_unavailable=True)
    client.indices.create(index="notebook_index", body=index_settings)


# initalize model
tokenizer = AutoTokenizer.from_pretrained("./all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("./all-MiniLM-L6-v2")

# Function to encode text into dense vectors
def encode_text(text):
    # Tokenize and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        output = model(**inputs)

    # Use the mean of the last hidden state as the sentence embedding
    embeddings = output.last_hidden_state.mean(dim=1).squeeze()

    # Convert to a numpy array
    return embeddings.numpy()


# Route to render HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# convert text file into data dictionary
def extract_file_content(file_path):
    extracted_data = {
        "class": "",
        "date": "",
        "topic": "",
        "notes": "",
        "topic_vector": None,
        "notes_vector": None
    }
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process lines
    lines = [line.strip() for line in lines]
    
    # extract data based on template
    extracted_data["class"] = lines[1]
    extracted_data["date"] = lines[4]
    extracted_data["topic"] = lines[7]
    extracted_data["notes"] = "\n".join(lines[10:])
    
    return extracted_data

# Route to handle file upload, extract text, and index in Elasticsearch
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("file")
    results = []

    for file in files:    
        if file.filename == '':
            results.append({"error": "No selected file"})
            continue

        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extract structured content from the file
        extracted_data = extract_file_content(file_path)

        # Remove the temporary file after extraction
        os.remove(file_path)

        # Encode the 'the notes and topic for vector search
        extracted_data["topic_vector"] = encode_text(extracted_data["topic"]).tolist()
        extracted_data["notes_vector"] = encode_text(extracted_data["notes"]).tolist()

        # confirm date formatting
        extracted_data["date"] = datetime.strptime(extracted_data["date"], "%Y-%m-%d").strftime("%Y-%m-%d")

        # Index the document in Elasticsearch
        response = client.index(index="notebook_index", body=extracted_data)
        results.append({"filename": file.filename, "id": response["_id"], "message": "Document added successfully"})
    
    return jsonify(results), 201

# Route to handle searching
@app.route('/search', methods=['POST'])
def search_documents():
    # get JSON payload from frontend
    data = request.json
    field = data.get("field")
    query = data.get("query")

    if not field or not query:
        return jsonify({"error": "Both field and query are required"}), 400
    
    response = None

    if field == "notes":
        response = client.search(
            index="notebook_index",
            knn={
                "field": "notes_vector",
                "query_vector": encode_text(query),
                "k": 10,
                "num_candidates": 100,
            },
        )
    elif field == "topic":
        response = client.search(
            index="notebook_index",
            knn={
                "field": "topic_vector",
                "query_vector": encode_text(query),
                "k": 10,
                "num_candidates": 100,
            },
        )
    elif field == "class":
        response = client.search(
            index="notebook_index",
            query={
                "match": {
                    "class": query
                }
            },
        )
    elif field == "date":
        start_date, end_date = query.split(" to ")
        response = client.search(
            index="notebook_index",
            query={
                "range": {
                    "date": {
                        "gte": start_date,
                        "lte": end_date,
                        "format": "yyyy-MM-dd"
                    }
                }
            },
        )
        
    return jsonify(response["hits"]["hits"])


# Run the app
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
