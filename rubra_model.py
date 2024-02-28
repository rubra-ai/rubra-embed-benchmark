import requests
import numpy as np

class RubraModel():
    def __init__(self):
        self.url = "http://localhost:8020/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key"
        }

    def encode(self, sentences, batch_size=32, **kwargs):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            # Prepare the batch of sentences
            batch_sentences = sentences[i:i+batch_size]
            # Prepare the data payload for the request
            data = {
                "input": batch_sentences,
                "model": "GPT-4",
                "encoding_format": "float"
            }
            # Make the POST request
            response = requests.post(self.url, json=data, headers=self.headers)
            # Check if the request was successful
            if response.status_code == 200:
                # Extract embeddings from the response
                response_data = response.json()
                if 'data' in response_data:
                    # Iterate over each item in the data and extract the embedding
                    for item in response_data['data']:
                        if 'embedding' in item:
                            embeddings.append(np.array(item['embedding']))
                else:
                    raise ValueError("Response JSON does not contain 'data'")
            else:
                # Raise an exception if the request was not successful
                raise Exception(f"Failed to get embeddings, status code: {response.status_code}")
        return embeddings
