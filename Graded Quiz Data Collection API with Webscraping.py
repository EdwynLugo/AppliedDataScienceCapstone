import requests
import pandas as pd
from bs4 import BeautifulSoup

# 1. Realizar la solicitud GET a la API de SpaceX
url = 'https://api.spacexdata.com/v4/launches'
response = requests.get(url)
data = response.json()

# Convertir la respuesta JSON a un DataFrame
df = pd.json_normalize(data)

# 2. Obtener el año de la primera fila en la columna 'static_fire_date_utc'
df['static_fire_date_utc'] = pd.to_datetime(df['static_fire_date_utc'], errors='coerce')
if pd.notna(df['static_fire_date_utc'].iloc[0]):
    first_row_year = df['static_fire_date_utc'].iloc[0].year
    print(f"Year in the first row (static_fire_date_utc): {first_row_year}")
else:
    print("No valid date found in 'static_fire_date_utc' for the first row.")

# 3. Contar los lanzamientos de Falcon 9 después de eliminar los de Falcon 1
# Primero verificamos que las columnas necesarias existan
if 'rocket' in df.columns:
    falcon_1_launches = df[df['rocket'].str.contains('Falcon 1', case=False, na=False)]
    falcon_9_launches = df[df['rocket'].str.contains('Falcon 9', case=False, na=False)]
    remaining_falcon_9 = len(falcon_9_launches) - len(falcon_1_launches)
    print(f"Remaining Falcon 9 launches after removing Falcon 1: {remaining_falcon_9}")
else:
    print("No 'rocket' column found in the DataFrame.")

# 4. Verificar los valores faltantes para 'landing_pad'
if 'landing_pad' in df.columns:
    missing_landing_pad = df['landing_pad'].isna().sum()
    print(f"Missing values for 'landing_pad': {missing_landing_pad}")
else:
    print("No 'landing_pad' column found in the DataFrame.")

# 5. Obtener el título de la página de Wikipedia
wiki_url = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"
wiki_response = requests.get(wiki_url)
soup = BeautifulSoup(wiki_response.text, 'html.parser')
wiki_title = soup.title.string if soup.title else "No title found"
print(f"Wiki page title: {wiki_title}")
