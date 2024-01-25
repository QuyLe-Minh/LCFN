import json
import numpy as np

# Load data from a JSON file
json_file_path = 'D:\LCFN\pretraining\hypergraph_embeddings.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Convert the data to NumPy arrays
user_embeddings = data[0]
item_embeddings = data[1]
