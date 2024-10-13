# ----------------------------------------------
# Code to create the database
# ----------------------------------------------
import os
import sqlite_utils

"""
Sets up and connects to the 'feedback_analysis.db' SQLite database, enabling data operations.

Attributes:
    db_path (str): Full path to the SQLite database file.
    db (sqlite_utils.Database): Connection object for database interactions.
"""

db_path = os.path.join(os.path.dirname(__file__), "feedback_analysis.db")
db = sqlite_utils.Database(db_path)
