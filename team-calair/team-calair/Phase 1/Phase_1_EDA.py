# Databricks notebook source
# MAGIC %pip install geopandas
# MAGIC %pip install shapely

# COMMAND ----------

DATA_BASE_DIR = "dbfs:/mnt/mids-w261/"
def connect_cloud_database () -> None:
    """
    Connects this notebook to the cloud database
    """

    # The following blob storage is accessible to team members only (read and write)
    # access key is valid til TTL
    # after that you will need to create a new SAS key and authenticate access again via DataBrick command line
    blob_container  = "ds261-final-project-team-2-2"       # The name of your container created in https://portal.azure.com
    storage_account = "ds261team"  # The name of your Storage account created in https://portal.azure.com
    secret_scope    = "261_team_2_2_scope"           # The name of the scope created in your local computer using the Databricks CLI
    secret_key      = "261_team_2_2_key"             # The name of the secret key created in your local computer using the Databricks CLI
    team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

    # https://ds261team.blob.core.windows.net/ds261-final-project-team-2-2

    # the 261 course blob storage is mounted here.
    mids261_mount_path      = "/mnt/mids-w261"

    # SAS Token: Grant the team limited access to Azure Storage resources
    spark.conf.set(
        f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
        dbutils.secrets.get(scope = secret_scope, key = secret_key)
    )

connect_cloud_database()

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame

def get_data_from_cloud () -> DataFrame:
    """
    Gets the first 3 months of data from the cloud storage
    """

    return spark.read.format("csv").option("header","true").load(f"{DATA_BASE_DIR}OTPW_3M_2015.csv")

df_flights_weather = get_data_from_cloud()

# COMMAND ----------

def show_metadata (df: DataFrame) -> None:
    """
    Displays the metadata of a DataFrame
    """

    print(f"{len(df.columns)} columns x {df.count()} rows")
    display(df)

show_metadata(df_flights_weather)

# COMMAND ----------

def find_empty_values_per_column (df: DataFrame) -> None:
    """
    Finds the number of empty values in every column
    """
    total = df.count()

    for column in df.columns:
        num_missing = df.filter(f'{column} IS NULL').count()
        percent = f"({(num_missing / total) * 100:.2f}%)" if num_missing > 0 else ""
        print(f"{column}: {num_missing} {percent}")

find_empty_values_per_column(df_flights_weather)

# COMMAND ----------

# Import shapefile for geopandas
!mkdir -p eda_data
!curl https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip -o eda_data/us_shapefiles.zip
!unzip eda_data/us_shapefiles.zip -d eda_data/us_shapefiles
!rm eda_data/us_shapefiles.zip

# COMMAND ----------

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def plot_usa_points (points: pd.DataFrame, title="") -> None:
    """
    Plots a list of latitute longitude pairs on a map of the USA

    Args:
    points (pd.DataFrame): List of latitute longitude pairs in a pandas DataFrame.
        The columns should be 'lat', 'lon', and optionally 'value' if you want to color the points.
        The data frame will be edited.
    title (str, optional): Title of plot. Defaults to "".
    """
    # Load shapefile and set map projection
    gdf = gpd.read_file('eda_data/us_shapefiles')

    points['lat'] = points['lat'].apply(pd.to_numeric)
    points['lon'] = points['lon'].apply(pd.to_numeric)
    if 'value' not in points.columns:
        points['value'] = 1

    point_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.lon, points.lat), crs="EPSG:4326")

    fig = plt.figure(1, figsize=(12,9)) 
    ax = fig.add_subplot()
    ax.set_xlim(-185, -63)
    ax.set_ylim(15, 75)
    ax.axis('off')

    if title != '':
        ax.set_title(title, fontsize=16)

    gdf.boundary.plot(ax=ax, color='black', linewidth=.6)
    point_gdf.plot(ax=ax, color='red', marker='o', markersize=8)

origin_airpoint_locations = df_flights_weather \
    .dropDuplicates(['ORIGIN_AIRPORT_ID']) \
    .selectExpr('origin_airport_lat as lat', 'origin_airport_lon as lon') \
    .toPandas()
plot_usa_points(origin_airpoint_locations, title='Origin Airport Locations')

# COMMAND ----------

