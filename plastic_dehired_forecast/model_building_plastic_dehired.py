import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read from SQLite
conn = sqlite3.connect('dehired_forecast.db')
df = pd.read_sql_query("SELECT * FROM raw_excel_data", conn)

# Preprocessing
df["ForecastedDate"] = pd.to_datetime(df["ForecastedDate"], format="%d-%m-%Y")
df["year"] = df["ForecastedDate"].dt.year
df["month"] = df["ForecastedDate"].dt.month
df["day"] = df["ForecastedDate"].dt.day
df = df.drop("ForecastedDate", axis=1)

labelencoder = LabelEncoder()
df['customername'] = labelencoder.fit_transform(df['customername'])

# Train/test data
X = df.iloc[:, :4]
y = df.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Serialize and store model
model_blob = pickle.dumps(model)

# Store model in DB
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS model_store (id INTEGER PRIMARY KEY AUTOINCREMENT, model BLOB)')
cursor.execute('DELETE FROM model_store')
cursor.execute('INSERT INTO model_store (model) VALUES (?)', (model_blob,))

# Store training data for potential inspection
cursor.execute('CREATE TABLE IF NOT EXISTS training_data (id INTEGER PRIMARY KEY AUTOINCREMENT, customername INTEGER, year INTEGER, month INTEGER, day INTEGER, predicted REAL)')
cursor.execute('DELETE FROM training_data')
df.to_sql('training_data', conn, if_exists='append', index=False)

conn.commit()
conn.close()
