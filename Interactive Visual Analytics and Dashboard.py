import folium
import pandas as pd
import wget
from folium.plugins import MarkerCluster
from folium.features import DivIcon
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

spacex_csv_file = wget.download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv')
spacex_df = pd.read_csv(spacex_csv_file)

spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]

nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

for _, row in launch_sites_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    folium.Marker(coordinate, icon=DivIcon(icon_size=(20,20), icon_anchor=(0,0), html=f'<div>{row["Launch Site"]}</div>')).add_to(site_map)

marker_cluster = MarkerCluster().add_to(site_map)
spacex_df['marker_color'] = spacex_df['class'].apply(lambda x: 'green' if x == 1 else 'red')

for _, row in spacex_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    folium.Marker(coordinate, icon=folium.Icon(color=row['marker_color'])).add_to(marker_cluster)

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

coastline_lat = 28.56367
coastline_lon = -80.57163

for _, row in launch_sites_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    distance = calculate_distance(row['Lat'], row['Long'], coastline_lat, coastline_lon)
    folium.Marker(coordinate, icon=folium.DivIcon(icon_size=(20,20), icon_anchor=(0,0), html=f'<div>{distance:.2f} km to Coastline</div>')).add_to(site_map)
    folium.PolyLine([coordinate, [coastline_lat, coastline_lon]], color="blue", weight=1).add_to(site_map)

if 'Payload Mass (kg)' not in spacex_df.columns:
    print("Warning: 'Payload Mass (kg)' column not found, check the CSV file.")
    spacex_df['Payload Mass (kg)'] = 0  

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SpaceX Launch Data"),
    html.Div([
        html.Label("Select Launch Site"),
        dcc.Dropdown(
            id='launch-site-dropdown',
            options=[{'label': site, 'value': site} for site in launch_sites_df['Launch Site']],
            value=launch_sites_df['Launch Site'][0]
        ),
    ]),
    dcc.Graph(id='success-pie-chart'),
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=100,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[0, 5000]
    ),
    dcc.Graph(id='payload-outcome-scatter')
])

@app.callback(
    Output('success-pie-chart', 'figure'),
    [Input('launch-site-dropdown', 'value')]
)
def update_pie_chart(launch_site):
    filtered_df = spacex_df[spacex_df['Launch Site'] == launch_site]
    success_counts = filtered_df['class'].value_counts()
    return px.pie(values=success_counts, names=['Failed', 'Successful'], title=f"Launch Outcome for {launch_site}")

@app.callback(
    Output('payload-outcome-scatter', 'figure'),
    [Input('payload-slider', 'value')]
)
def update_scatter_plot(payload):
    filtered_df = spacex_df[spacex_df['Payload Mass (kg)'] <= payload[1]]
    return px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Launch Site', title="Payload vs Outcome")

if __name__ == '__main__':
    app.run_server(debug=True)
