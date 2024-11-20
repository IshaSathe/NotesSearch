# Notes Search
> An app that allows you to upload and search notes

## Capabilities
- Upload notes as a plain text file in the following format
```txt
Class
Class Name

Date
YYY-MM-DD

Topic
Topic Name

Notes
  - some random note 1
  - some random note 2 ....
```
- Can search through files by:
  - date: in the format "YYYY-MM-DD to YYYY-MM-DD"
  - class: finds an exact match to class name
  - topic: uses vector search on topics
  - all notes: uses vector seach on notes

## Walkthrough
1. Install App
   1. clone repo
   2. run `docker-compose build --no-cache`
   3. run `docker compose up`
   4. navigate to elasticsearch `http://localhost:9200/`
   5. navigate to app `http://127.0.0.1:5000/`
2. Upload Data
   1. test data is located in the 'test data' folder, it contains 30 notes on different subjects
   2. upload all these files
3. Search
   1. When creating the index the following mappings were used
      ```python
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
      ````
   2. The date search uses a range query
   3. The class search match query
   4. The topic and notes search use vector search
      - Uses a more lightweight approach than `SentenceTransformers` (created large load times during image creation)
      - Dowloaded the files for the `all-MiniLM-L6-v2` model from Hugging Face
      - Uses the `tranformers` library
        - `AutoTokenizer` handles tokenization, splitting text into tokens that the model can understand
        - `Automodel` loads the transformer model with its pre-trained weights
      - Directly pass text through the model to generate embeddings
        ```python
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
        ```





