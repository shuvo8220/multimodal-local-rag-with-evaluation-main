import requests

def get_embedding(prompt, model="nomic-embed-text"):

    url = "http://localhost:11434/api/embeddings/"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "model": model}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("embedding", [])
    else:
        raise Exception(
            f"Error fetching embedding: {response.status_code}, {response.text}"
        )

