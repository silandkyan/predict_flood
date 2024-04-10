import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import rasterio
#from rasterio.plot import show
#import cartopy.crs as ccrs # probably needs to be installed with pip...
from shapely.geometry import Point

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA


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
    #w = calc_cumulative_weather(w)

    return w


def calc_cumulative_weather(df):
    windows = {'2d': 2, '7d': 7,
                '30d': 30, '90d': 90,
                '1y': 365, '3y': 365*3}

    for w in windows:
        df[f'precip_mean_prev_{w}_sum'] = df['precip_mean'].rolling(window=windows[w]).sum()
        df[f'tmean_mean_prev_{w}_mean'] = df['tmean_mean'].rolling(window=windows[w]).mean()

    return df


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

    initial_years_agg = initial_years_agg.rename(columns={'water_depth_mean': 'ini_years_water_depth_mean',
                                                        'water_depth_std': 'ini_years_water_depth_std',
                                                        'water_depth_min': 'ini_years_water_depth_min',
                                                        'water_depth_max': 'ini_years_water_depth_max'})

    stations_agg = gpd.GeoDataFrame(pd.merge(stations, initial_years_agg, on='station_id', how='left'),
                                    geometry='geometry')

    return stations_agg


def merge_groundwater_data(data, stations):
    merged = pd.merge(data, stations, how='left')
    merged['water_depth_anomaly'] = merged['water_depth'] - merged['ini_years_water_depth_mean']
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


def calc_residuals(y_test, y_pred):
    resid = pd.DataFrame()
    resid['observed'] = y_test.copy()
    resid['predicted'] = y_pred.copy()
    resid['residuals'] = resid['predicted'] - resid['observed']
    return resid


def perform_pca(df, n_components=None):
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    # Perform PCA
    if n_components is None:
        n_components = min(df.shape[1], df.shape[0])
    pca = PCA(n_components=n_components).set_output(transform='pandas')
    X_pca = pca.fit_transform(data_scaled)
    
    # Gather PCA statistics
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    
    # Create a DataFrame for explained variance ratio
    df_expl_vari_ratio = pd.DataFrame(explained_variance_ratio,
                                      columns=['Explained Variance Ratio'],
                                      index=[f'PC-{i+1}' for i in range(n_components)])
    
    # Create a DataFrame for components
    df_components = pd.DataFrame(components, 
                                 columns=df.columns, 
                                 index=[f'PC-{i+1}' for i in range(n_components)])
    
    # Concatenate DataFrames
    df_pca_stats = pd.concat([df_expl_vari_ratio, df_components], axis=1)
    
    return pca, X_pca, df_pca_stats.T


def explore_clusters(df, scale=True):
    # DATA SCALING
    if scale == True:
        # Initialise the transformer (optionally, set parameters)
        min_max = MinMaxScaler().set_output(transform="pandas")
        
        # Use the transformer to transform the data
        df = min_max.fit_transform(df)

    wcss = []
    clusters = []
    silhouettes = []

    # CLUSTERING
    for k in range(2, 30):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        n_init=10, max_iter=300)
        
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(df, kmeans.labels_))
        clusters.append(k)

    slope = np.diff(wcss)
    curve = np.diff(slope)
    #elbow = np.argmax(curve) + 1
    #print(clusters[elbow])

    # PLOTTING
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0,0].plot(clusters, wcss)
    axs[0,0].set_xlabel('# of clusters')
    axs[0,0].set_ylabel('inertia')
    
    axs[0,1].plot(clusters, silhouettes)
    axs[0,1].set_xlabel('# of clusters')
    axs[0,1].set_ylabel('silhouette score')
    
    axs[1,0].plot(clusters[:-1], slope)
    axs[1,0].set_xlabel('# of clusters')
    axs[1,0].set_ylabel('inertia slope')
    
    axs[1,1].plot(clusters[:-2], curve)
    axs[1,1].set_xlabel('# of clusters')
    axs[1,1].set_ylabel('inertia curvature')


def apply_clusters(df, n_clusters, scale=True):
    # DATA SCALING
    if scale == True:
        # Initialise the transformer (optionally, set parameters)
        min_max = MinMaxScaler().set_output(transform="pandas")
        
        # Use the transformer to transform the data
        df = min_max.fit_transform(df)
        
     # CLUSTERING
    kmeans = KMeans(n_clusters=n_clusters, #random_state=0,
                    n_init='auto').fit(df)

    return kmeans.labels_, kmeans.cluster_centers_


def plot_clusters(coordinates_df, labels, centers):
    '''
    coordinates_df of the form: df[['x', 'y']]
    '''
    # Plot the points
    plt.scatter(coordinates_df['x'], coordinates_df['y'], c=labels)#, cmap='viridis')
    
    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    
    # Show the plot
    plt.show()
    
    
def tt_split_by_stations(df):
    ids = list(df.station_id.unique())

    # Calculate 80% of the list's length
    num_elements = round(len(ids) * 0.8)
    
    # Randomly select 80% of the elements
    train_ids = random.sample(ids, k=num_elements)

    # create train and test dfs
    train = df.loc[df.station_id.isin(train_ids)]
    test = df.loc[~df.station_id.isin(train_ids)]

    # define X and y
    y_train = train.pop('water_depth')
    y_test = test.pop('water_depth')
    
    reserve_cols_as_info = ['station_id', 'date', 'geometry']
    
    info_train = train[reserve_cols_as_info].copy()
    info_test = test[reserve_cols_as_info].copy()
    
    X_train = train.copy().drop(reserve_cols_as_info, axis=1)
    X_test = test.copy().drop(reserve_cols_as_info, axis=1)

    return train, test, X_train, X_test, y_train, y_test, info_train, info_test


def plot_station_data(df, start_date=None, end_date=None):
    # Create a figure and an axis
    fig, ax = plt.subplots(4,1, figsize=(10, 10), sharex=True)
    
    sns.lineplot(x="date", y="water_depth",
                 color='tab:orange', alpha=0.5,
                 data=df, ax=ax[0], label='Measured (m)')
    
    sns.lineplot(x="date", y="pred_water_depth",
                 color='tab:red', alpha=0.5,
                 data=df, ax=ax[0], label='Predicted (m)')
    
    ax[0].legend()
    
    sns.lineplot(x="date", y="residuals",
                 color='k', alpha=0.5,
                 data=df, ax=ax[1])
    
    ax[1].axhline(y=0, color='k', alpha=0.7)
    # Create a secondary y-axis
    #ax[1] = ax[0].twinx()
    
    sns.lineplot(x="date", y="tmean_mean_prev_1y_mean", 
                 data=df, ax=ax[2], color='tab:green',
                 label='Mean Temp. prev. year (°C)')
    
    # Create a secondary y-axis
    ax[2] = ax[2].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_1y_sum", 
                 data=df, ax=ax[2], color='tab:blue',
                 label='Cumulative precip. prev. year (mm)')
    
    ax[2].legend()
    
    # Optionally, set labels for the y-axes
    #ax[3].set_ylabel('precip')
    #ax2.set_ylabel('Y2 Label')
    
    # Plot the second DataFrame on the secondary y-axis
    sns.lineplot(x="date", y="tmean_mean_prev_7d_mean", 
                 data=df, ax=ax[3], color='tab:green',
                 label='Mean temp. prev. year (°C)')
    
    # Create a secondary y-axis
    ax[3] = ax[3].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_7d_sum", 
                 data=df, ax=ax[3], color='tab:blue',
                 label='Cumulative precip. prev. 7 days (mm)')
    
    ax[3].legend()
    
    # Optionally, set labels for the y-axes
    #ax[3].set_ylabel('precip')
    #ax2.set_ylabel('Y2 Label')
    
    # Set the x-axis limits
    if start_date and end_date:
        plt.xlim(start_date, end_date)
    
    # Show the plot
    plt.show()
