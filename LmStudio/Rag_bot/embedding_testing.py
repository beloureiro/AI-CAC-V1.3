# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding = model.encode(["apple"])
# print(embedding)


# import os
# import sqlite_utils

# # Define o caminho completo para o banco de dados
# db_path = os.path.join(os.path.dirname(__file__), "feedback_analysis.db")
# db = sqlite_utils.Database(db_path)


import pandas as pd
import sqlite_utils
import os

# Define o caminho para o banco de dados SQLite
db_path = os.path.join(os.path.dirname(__file__), "feedback_analysis.db")
db = sqlite_utils.Database(db_path)

# Carrega o arquivo Excel
excel_file = "TableSchema.xlsx"
data = pd.ExcelFile(os.path.join(os.path.dirname(__file__), excel_file))

# Define as tabelas com as abas do Excel (exceto "Instruction")
table_sheets = {
    "Feedback": "Feedback",
    "PatientExperienceExpert": "PatientExperienceExpert",
    "HealthITProcessExpert": "HealthITProcessExpert",
    "ClinicalPsychologist": "ClinicalPsychologist",
    "CommunicationExpert": "CommunicationExpert",
    "ManagerAndAdvisor": "ManagerAndAdvisor"
}

# Cria as tabelas e insere os dados
for sheet_name, table_name in table_sheets.items():
    # Lê os dados da aba específica
    df = data.parse(sheet_name)
    
    # Insere os dados na tabela SQLite, substituindo se já existir
    db[table_name].insert_all(df.to_dict(orient="records"), pk="Feedback_ID", replace=True)

print("Tabelas criadas e dados inseridos com sucesso.")
