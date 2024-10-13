# ----------------------------------------------
# Code to create the master table
# ----------------------------------------------
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
Creates a consolidated master table by merging data from existing tables and generating embeddings for text columns.

This script:
1. Connects to the SQLite database.
2. Loads a SentenceTransformer model for embedding generation.
3. Merges all tables except the consolidated one into a master DataFrame.
4. Generates embeddings for text columns and saves the consolidated table back to the database.

Functions:
    generate_embedding(text): Generates embeddings for a given text using the loaded model.
    create_consolidated_table(): Creates and saves the consolidated table, including generated embeddings for text columns.
"""

# Connect to the SQLite database
conn = sqlite3.connect('LmStudio/Rag_bot/feedback_analysis.db')

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embedding
def generate_embedding(text):
    return model.encode(str(text), show_progress_bar=False)

# Function to create the consolidated table
def create_consolidated_table():
    # Read all existing tables except the consolidated table
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name != 'aicac_consolidated';", conn)
    
    # Initialize the consolidated DataFrame
    consolidated_df = pd.DataFrame()
    
    for table_name in tables['name']:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Add to the consolidated DataFrame
        if consolidated_df.empty:
            consolidated_df = df
        else:
            consolidated_df = pd.merge(consolidated_df, df, on='Feedback_ID', how='outer', suffixes=('', '_drop'))
            # Remove duplicate columns
            consolidated_df = consolidated_df.loc[:, ~consolidated_df.columns.str.endswith('_drop')]
    
    # Generate embeddings for each text column
    for column in consolidated_df.columns:
        if consolidated_df[column].dtype == 'object' and not column.endswith('_embedding'):
            consolidated_df[f"{column}_embedding"] = consolidated_df[column].apply(generate_embedding)
    
    # Save the consolidated table to the database
    consolidated_df.to_sql('aicac_consolidated', conn, if_exists='replace', index=False)
    print("aicac_consolidated table created successfully.")
    
    # Print columns for verification
    print("Columns in the aicac_consolidated table:")
    print(consolidated_df.columns.tolist())

# Execute the function
create_consolidated_table()

# Close the connection
conn.close()
