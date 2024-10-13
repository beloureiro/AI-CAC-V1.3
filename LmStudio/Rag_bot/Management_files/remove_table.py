# ----------------------------------------------
# Code to remove the 'consolidated_feedback' table
# ----------------------------------------------
import sqlite3

"""
Connects to an SQLite database and deletes the 'consolidated_feedback' table if it exists.

This script:
1. Connects to the specified SQLite database.
2. Checks if the 'consolidated_feedback' table exists.
3. Drops the table if it exists, or prints a message if it does not.
4. Commits changes and closes the database connection.
"""

# Connect to the SQLite database
conn = sqlite3.connect('LmStudio/Rag_bot/feedback_analysis.db')

# Create a cursor
cursor = conn.cursor()

# Check if the table exists before attempting to drop it
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='consolidated_feedback';")
if cursor.fetchone():
    # If the table exists, drop it
    cursor.execute("DROP TABLE consolidated_feedback;")
    print("Table 'consolidated_feedback' removed successfully.")
else:
    print("The table 'consolidated_feedback' does not exist in the database.")

# Commit changes and close the connection
conn.commit()
conn.close()
