import pandas as pd
import sqlite3

# Load Excel data
df = pd.read_excel('plastic_dehired_model.xlsx')

# Connect to SQLite DB
conn = sqlite3.connect('dehired_forecast.db')

# Create raw table and upload data
df.to_sql('raw_excel_data', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print("Excel data uploaded to SQLite successfully.")
