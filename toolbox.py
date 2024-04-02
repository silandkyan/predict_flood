import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.plot import show
import cartopy.crs as ccrs # probably needs to be installed with pip...
import pyproj
from shapely.geometry import Point

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


def load_dem(file):
    with rasterio.open(file) as src:
        dem = src.read(1)
        extent = src.bounds

    return dem, extent


def load_weather_station_data(file):
    stations = pd.read_csv(file)
    
    # Create a geometry column with Point objects
    geometry = [Point(x, y) for x, y in zip(stations['xcoord'], stations['ycoord'])]

    # Create a GeoDataFrame from the DataFrame and geometry column
    stations = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:4326')

    # Transform to datetime
    stations.start_date = pd.to_datetime(stations.start_date)
    stations.end_date = pd.to_datetime(stations.end_date)
    
    return stations


def load_weather_data(file):
    w = pd.read_csv(file)

    # change date column to datetime type
    w.date = pd.to_datetime(w.date)
    w.set_index('date', inplace=True)

    # calculate cumulative weather data for moving windows
    w['precip_mean_prev_7d_sum'] = w['precip_mean'].rolling(window=7).sum()
    w['precip_mean_prev_30d_sum'] = w['precip_mean'].rolling(window=30).sum()
    w['precip_mean_prev_90d_sum'] = w['precip_mean'].rolling(window=90).sum()    
    w['tmean_mean_prev_7d_mean'] = w['tmean_mean'].rolling(window=7).mean()
    w['tmean_mean_prev_30d_mean'] = w['tmean_mean'].rolling(window=30).mean()
    w['tmean_mean_prev_90d_mean'] = w['tmean_mean'].rolling(window=90).mean()

    return w


def load_groundwater_station_data(file):
    stations = pd.read_csv(file)
    
    # correct data types
    stations.start_date = pd.to_datetime(stations.start_date)
    stations.end_date = pd.to_datetime(stations.end_date)
    stations.lifespan = pd.to_timedelta(stations.lifespan)

    # Create a geometry column with Point objects
    geometry = [Point(x, y) for x, y in zip(stations['x'], stations['y'])]

    # Create a GeoDataFrame from the DataFrame and geometry column
    stations = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:32632')
    stations = stations.to_crs('EPSG:4326')
    
    return stations


def load_groundwater_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])

    return df


def calc_aggregate_station_data(stations, data):
    aggregated = data.groupby('station_id').agg({'water_level': 'mean',
                                             'water_depth': 'mean',
                                             'water_temp': 'mean'}).reset_index()

    aggregated = aggregated.rename(columns={'water_level': 'water_level_mean',
                                             'water_depth': 'water_depth_mean',
                                             'water_temp': 'water_temp_mean'})

    stations_agg = pd.merge(stations, aggregated, on='station_id', how='left')

    return stations_agg


def calc_initial_station_data(stations, data, initial_n_years):
    df_list = []

    for id in data.station_id.unique():
        df = data[data.station_id == id]
        start_date = df.date.min()
        end_date = start_date + pd.Timedelta(days=365*initial_n_years)
        df_first_year = df.loc[(df['date'] >= start_date) & (df['date'] < end_date)]
        df_list.append(df_first_year)

    initial_years = pd.concat(df_list)
    initial_years_agg = initial_years.groupby('station_id').agg({'water_depth': ['mean',
                                                                                'std',
                                                                                'min',
                                                                                'max']}
                                                                ).reset_index()

    initial_years_agg.columns = ["_".join(col).rstrip('_') for col in initial_years_agg.columns.values]

    initial_years_agg = initial_years_agg.rename(columns={'water_depth_mean': f'ini_{initial_n_years}_years_water_depth_mean',
                                                        'water_depth_std': f'ini_{initial_n_years}_years_water_depth_std',
                                                        'water_depth_min': f'ini_{initial_n_years}_years_water_depth_min',
                                                        'water_depth_max': f'ini_{initial_n_years}_years_water_depth_max'})

    stations_agg = gpd.GeoDataFrame(pd.merge(stations, initial_years_agg, on='station_id', how='left'),
                                    geometry='geometry')

    return stations_agg


def merge_groundwater_data(data, stations):
    merged = pd.merge(data, stations, how='left')
    merged['water_depth_anomaly'] = merged['water_depth'] - merged['ini_1_years_water_depth_mean']
    merged.index = merged['date']

    return merged


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true = y_true,
                              y_pred = y_pred)
    
    rmse = mean_squared_error(y_true = y_true,
                              y_pred = y_pred,
                              squared=False)

    mape = mean_absolute_percentage_error(y_true = y_true,
                                          y_pred = y_pred)
    
    r2 = r2_score(y_true = y_true, y_pred = y_pred)

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
    
    print('MAE', mae)
    print('RMSE', rmse)
    print('MAPE', mape)
    print('R2', r2)
    
    return metrics


