import sqlite3

# Conectar ao banco de dados SQLite
conn = sqlite3.connect('LmStudio/Rag_bot/feedback_analysis.db')

# Criar um cursor
cursor = conn.cursor()

# Verificar se a tabela existe antes de tentar removê-la
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='consolidated_feedback';")
if cursor.fetchone():
    # Se a tabela existe, removê-la
    cursor.execute("DROP TABLE consolidated_feedback;")
    print("Tabela 'consolidated_feedback' removida com sucesso.")
else:
    print("A tabela 'consolidated_feedback' não existe no banco de dados.")

# Commit as mudanças e fechar a conexão
conn.commit()
conn.close()