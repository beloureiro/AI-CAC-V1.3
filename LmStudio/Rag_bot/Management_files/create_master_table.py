import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer

# Conectar ao banco de dados SQLite
conn = sqlite3.connect('LmStudio/Rag_bot/feedback_analysis.db')

# Carregar o modelo de embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Função para gerar embedding
def generate_embedding(text):
    return model.encode(str(text), show_progress_bar=False)

# Função para criar a tabela consolidada
def create_consolidated_table():
    # Ler todas as tabelas existentes, excluindo a tabela consolidada
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name != 'aicac_consolidated';", conn)
    
    # Inicializar o DataFrame consolidado
    consolidated_df = pd.DataFrame()
    
    for table_name in tables['name']:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Adicionar ao DataFrame consolidado
        if consolidated_df.empty:
            consolidated_df = df
        else:
            consolidated_df = pd.merge(consolidated_df, df, on='Feedback_ID', how='outer', suffixes=('', '_drop'))
            # Remover colunas duplicadas
            consolidated_df = consolidated_df.loc[:, ~consolidated_df.columns.str.endswith('_drop')]
    
    # Gerar embeddings para cada coluna de texto
    for column in consolidated_df.columns:
        if consolidated_df[column].dtype == 'object' and not column.endswith('_embedding'):
            consolidated_df[f"{column}_embedding"] = consolidated_df[column].apply(generate_embedding)
    
    # Salvar a tabela consolidada no banco de dados
    consolidated_df.to_sql('aicac_consolidated', conn, if_exists='replace', index=False)
    print("Tabela aicac_consolidated criada com sucesso.")
    
    # Imprimir as colunas para verificação
    print("Colunas na tabela aicac_consolidated:")
    print(consolidated_df.columns.tolist())

# Executar a função
create_consolidated_table()

# Fechar a conexão
conn.close()