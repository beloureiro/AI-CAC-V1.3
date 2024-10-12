import sqlite3

# Caminho para o banco de dados
db_path = "LmStudio/Rag_bot/feedback_analysis.db"

# Conectar ao banco de dados SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Executar uma consulta para listar as colunas da tabela
table_name = "consolidated_feedback"
cursor.execute(f"PRAGMA table_info({table_name});")

# Recuperar e exibir os nomes das colunas
columns = [column[1] for column in cursor.fetchall()]
print("Colunas da tabela:", columns)

# Fechar a conexão com o banco de dados
cursor.close()
conn.close()
