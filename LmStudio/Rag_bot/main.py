import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class RAGBot:
    """
    RAGBot is a Retrieval-Augmented Generation chatbot that interacts with a large text dataset.
    It loads and consolidates data, splits it into chunks, generates embeddings, stores them in a FAISS index, 
    and retrieves the most relevant chunks based on user queries. It also calls an external LLM (from LM Studio) 
    to generate context-based responses.
    """

    LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

    def __init__(self, data_directory=None, consolidated_path='consolidated_text.txt', chunk_size=500):
        """
        Initializes the RAGBot with default parameters.
        
        :param data_directory: Directory containing Markdown files to be loaded. If not provided, defaults to the correct path.
        :param consolidated_path: Path to save the consolidated text file.
        :param chunk_size: Size of chunks to split the text into for processing.
        """
        if data_directory is None:
            data_directory = os.path.join(os.path.dirname(__file__), '..', 'lms_reports_md')
        
        self.data_directory = os.path.abspath(data_directory)
        self.consolidated_path = consolidated_path
        self.chunk_size = chunk_size
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []

    def load_data(self):
        """
        Loads and consolidates all Markdown (.md) files in the specified directory into a single text string.
        Saves the consolidated text to a file.
        """
        all_text = ""
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".md"):
                file_path = os.path.join(self.data_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_text += file.read() + "\n"  # Add newline between files for clarity

        # Save the consolidated text to a file
        with open(self.consolidated_path, 'w', encoding='utf-8') as outfile:
            outfile.write(all_text)

        print(f"Consolidated text saved to {self.consolidated_path}")
        return all_text

    def split_text_into_chunks(self, text):
        """
        Splits the input text into smaller chunks based on the specified chunk size.
        
        :param text: The full consolidated text to be split.
        :return: A list of text chunks.
        """
        self.chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        print(f"Text has been split into {len(self.chunks)} chunks.")
        return self.chunks

    def generate_embeddings(self):
        """
        Generates embeddings for the text chunks using a pre-trained model.
        
        :return: A list of embeddings corresponding to the text chunks.
        """
        embeddings = self.model.encode(self.chunks)
        print("Generated embeddings for the text chunks.")
        return embeddings

    def build_faiss_index(self, embeddings):
        """
        Builds a FAISS index using the provided embeddings.
        
        :param embeddings: A list of embeddings to store in the FAISS index.
        """
        embedding_dim = embeddings[0].shape[0]  # Dimensionality of embeddings
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance-based FAISS index
        self.index.add(np.array(embeddings))  # Add embeddings to the index
        print("FAISS index built with embeddings.")

    def retrieve_similar_chunks(self, query):
        """
        Retrieves the top-k most similar chunks to the query embedding from the FAISS index.
        
        :param query: The user's query string.
        :param k: The number of top results to return (default is 5).
        :return: The most relevant text chunks.
        """
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), 5)  # Search for 5 similar chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        print(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        return relevant_chunks

    def call_llm(self, query, context_chunks, structured=False):
        """
        Calls the LM Studio model with the user's query and relevant context chunks.
        Optionally supports structured output if requested.
        
        :param query: User's input query.
        :param context_chunks: The retrieved text chunks relevant to the query.
        :param structured: Boolean to indicate if structured output is requested.
        :return: The response from the LLM.
        """
        # Define the system message to pass the LLM its identity and role
        system_message = {
            "role": "system", 
            "content": "You are a helpful assistant that provides accurate and relevant responses based on the provided context."
        }

        # The user's query
        user_message = {"role": "user", "content": query}

        # Context combined with the question
        prompt = f"Context:\n{''.join(context_chunks)}\n\nQuestion: {query}\nAnswer:"
        
        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": [
                system_message,   # Pass the system message here
                {"role": "user", "content": prompt}  # The user's prompt and context
            ],
            "temperature": 0.7,
            "max_tokens": 200,
            "stream": False
        }
        
        # If structured output is requested, add the response_format schema
        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "additional_info": {"type": "string"}
                        },
                        "required": ["answer"]
                    }
                }
            }

        # Send the request to LM Studio
        response = requests.post(self.LM_STUDIO_API_URL, json=payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error: Failed to get response from LM Studio."


    def chatbot(self):
        """
        Runs the chatbot, which loads data, generates embeddings, builds a FAISS index, and interacts with the user.
        """
        # Step 1: Load and consolidate Markdown files
        text = self.load_data()

        # Step 2: Split text into chunks
        self.split_text_into_chunks(text)

        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings()

        # Step 4: Build FAISS index
        self.build_faiss_index(embeddings)

        # Step 5: Chat interaction with user
        print("Chatbot is ready. Type your question below.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            # Step 6: Retrieve relevant chunks and call LLM
            relevant_chunks = self.retrieve_similar_chunks(user_input)
            llm_response = self.call_llm(user_input, relevant_chunks)
            print(f"LLM Response: {llm_response}")


if __name__ == "__main__":
    bot = RAGBot()
    bot.chatbot()
