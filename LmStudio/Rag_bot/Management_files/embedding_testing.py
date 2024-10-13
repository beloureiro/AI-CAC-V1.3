# ----------------------------------------------
# Code to test embedding generation
# ----------------------------------------------
from sentence_transformers import SentenceTransformer

"""
Tests sentence embedding generation using the SentenceTransformer model.

This script:
1. Loads the 'all-MiniLM-L6-v2' model from the SentenceTransformer library.
2. Encodes the word "apple" to generate its embedding.
3. Prints the resulting embedding to the console.
"""

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embedding for the word "apple"
embedding = model.encode(["apple"])

# Print the generated embedding
print(embedding)
