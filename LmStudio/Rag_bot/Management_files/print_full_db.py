# ----------------------------------------------
# Code to print the full database tables and columns names
# ----------------------------------------------
import sqlite3

# Caminho para o banco de dados
db_path = "LmStudio/Rag_bot/feedback_analysis.db"

# Conectar ao banco de dados SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Consultar todas as tabelas no banco de dados
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Iterar sobre cada tabela e listar suas colunas
for table in tables:
    table_name = table[0]
    print(f"\nTabela: {table_name}")

    # Consultar as colunas da tabela atual
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [column[1] for column in cursor.fetchall()]
    print("Colunas:", columns)

# Fechar a conexão com o banco de dados
cursor.close()
conn.close()
