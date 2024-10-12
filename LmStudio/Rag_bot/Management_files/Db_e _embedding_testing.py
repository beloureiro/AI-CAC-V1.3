# ----------------------------------------------
# codigo para testar o embedding
# ----------------------------------------------
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding = model.encode(["apple"])
# print(embedding)

# ----------------------------------------------
# codigo para o banco de dados
# ----------------------------------------------
# import os
# import sqlite_utils

# # Define o caminho completo para o banco de dados
# db_path = os.path.join(os.path.dirname(__file__), "feedback_analysis.db")
# db = sqlite_utils.Database(db_path)

# ----------------------------------------------
# codigo para criar as tabelas no banco de dados a partir do excel
# ----------------------------------------------
import pandas as pd
import sqlite_utils
import os

# Diretório base do projeto
base_dir = r"D:\OneDrive - InMotion - Consulting\AI Projects\AI-CAC-V1.3"

# Define o caminho para o banco de dados SQLite
db_path = os.path.join(base_dir, "LmStudio", "Rag_bot", "feedback_analysis.db")
# Verifica se o arquivo de banco de dados existe
if not os.path.exists(db_path):
    raise FileNotFoundError(f"O arquivo de banco de dados não foi encontrado em: {db_path}")

db = sqlite_utils.Database(db_path)

# Define o caminho completo para o arquivo Excel
excel_file = os.path.join(base_dir, "LmStudio", "Rag_bot", "Management_files", "TableSchema.xlsx")
# Verifica se o arquivo Excel existe
if not os.path.exists(excel_file):
    raise FileNotFoundError(f"O arquivo Excel não foi encontrado em: {excel_file}")

# Carrega o arquivo Excel
data = pd.ExcelFile(excel_file)

# Define as tabelas a serem recriadas (excluindo "Instruction")
table_sheets = {
    "Feedback": "Feedback",
    "PatientExperienceExpert": "PatientExperienceExpert",
    "HealthITProcessExpert": "HealthITProcessExpert",
    "ClinicalPsychologist": "ClinicalPsychologist",
    "CommunicationExpert": "CommunicationExpert",
    "ManagerAndAdvisor": "ManagerAndAdvisor"
}

# Remove as tabelas e recria de acordo com o Excel
for sheet_name, table_name in table_sheets.items():
    # Lê os dados da aba específica
    df = data.parse(sheet_name)

    # Remove a tabela se ela já existir
    if table_name in db.table_names():
        db[table_name].drop()

    # Cria uma nova tabela e insere os dados
    db[table_name].insert_all(df.to_dict(orient="records"), pk="Feedback_ID", replace=True)

print("Tabelas recriadas e dados inseridos com sucesso.")
