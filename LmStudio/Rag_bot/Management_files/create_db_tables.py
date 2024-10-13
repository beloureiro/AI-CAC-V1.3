# ----------------------------------------------
# Code to create the tables
# ----------------------------------------------
import pandas as pd
import sqlite_utils
import os

"""
Creates tables in an SQLite database from an Excel file and inserts data into the tables.

This script:
1. Establishes the project base directory and paths for the SQLite database and Excel file.
2. Verifies the existence of the database and Excel file, raising an error if either is missing.
3. Loads the Excel file and recreates specified tables based on the Excel sheet names.
4. Drops existing tables and recreates them with new data from the Excel file.

Attributes:
    base_dir (str): The base directory path for the project files.
    db_path (str): The full path to the SQLite database file.
    excel_file (str): The full path to the Excel file containing table data.
    table_sheets (dict): Mapping of Excel sheet names to SQLite table names.
"""

# Base directory of the project
base_dir = r"D:\OneDrive - InMotion - Consulting\AI Projects\AI-CAC-V1.3"

# Define the path to the SQLite database
db_path = os.path.join(base_dir, "LmStudio", "Rag_bot", "feedback_analysis.db")
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found at: {db_path}")

db = sqlite_utils.Database(db_path)

# Define the full path to the Excel file
excel_file = os.path.join(base_dir, "LmStudio", "Rag_bot", "Management_files", "TableSchema.xlsx")
if not os.path.exists(excel_file):
    raise FileNotFoundError(f"Excel file not found at: {excel_file}")

# Load the Excel file
data = pd.ExcelFile(excel_file)

# Define tables to recreate (excluding "Instruction")
table_sheets = {
    "Feedback": "Feedback",
    "PatientExperienceExpert": "PatientExperienceExpert",
    "HealthITProcessExpert": "HealthITProcessExpert",
    "ClinicalPsychologist": "ClinicalPsychologist",
    "CommunicationExpert": "CommunicationExpert",
    "ManagerAndAdvisor": "ManagerAndAdvisor"
}

# Remove and recreate tables based on Excel
for sheet_name, table_name in table_sheets.items():
    df = data.parse(sheet_name)

    # Drop the table if it already exists
    if table_name in db.table_names():
        db[table_name].drop()

    # Create a new table and insert data
    db[table_name].insert_all(df.to_dict(orient="records"), pk="Feedback_ID", replace=True)

print("Tables recreated and data successfully inserted.")
