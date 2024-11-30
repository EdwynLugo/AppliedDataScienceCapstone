import sqlite3
import pandas as pd
import requests
import json

conn = sqlite3.connect('spacex.db')
cursor = conn.cursor()

spacex_url = "https://api.spacexdata.com/v4/launches/past"
data = pd.json_normalize(requests.get(spacex_url).json())

for column in data.columns:
    if data[column].apply(lambda x: isinstance(x, (list, dict))).any():
        data[column] = data[column].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

data.to_sql('spacex_data', conn, if_exists='replace', index=False)
print("SpaceX dataset loaded into SQLite table.")

query = "SELECT * FROM spacex_data LIMIT 10;"
df = pd.read_sql_query(query, conn)
print(df)

conn.close()
