import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN
from datetime import datetime

# Load the dataset
data = pd.read_csv('code1_dataset.csv')

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract year and month for temporal analysis
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Prepare data for clustering: combining spatial and temporal data
data['YearMonth'] = data['Year'].astype(str) + '-' + data['Month'].astype(str)
data_for_clustering = data[['Latitude', 'Longitude', 'Year', 'Month']]

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=2)
clusters = dbscan.fit_predict(data_for_clustering)
   
# Add cluster labels to the dataset
data['Cluster'] = clusters

# Visualize the clusters
fig = px.scatter_mapbox(
    data, lat='Latitude', lon='Longitude', color='Cluster', size='Cases',
    hover_data={'Latitude': False, 'Longitude': False, 'Cases': True, 'Date': True, 'Region': True},
    title="Dengue Cases Clustering",
    mapbox_style="carto-positron"
)

fig.update_layout(mapbox=dict(zoom=5, center=dict(lat=3.0, lon=101.5)))
fig.show()
