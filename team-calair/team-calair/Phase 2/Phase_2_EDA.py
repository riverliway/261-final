# Databricks notebook source
# MAGIC %pip install geopandas
# MAGIC %pip install shapely
# MAGIC %pip install geoplot

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
from typing import Literal

def get_data_from_cloud (dataset: Literal['OTPW_3M_2015.csv', 'OTPW_12M/OTPW_12M_2015.csv.gz']) -> DataFrame:
    """
    Gets the dataset from the cloud
    """

    return spark.read.format("csv").option("header","true").load(f"{DATA_BASE_DIR}{dataset}")

df_flights_weather_3m = get_data_from_cloud('OTPW_3M_2015.csv')

# COMMAND ----------

def show_metadata (df: DataFrame) -> None:
    """
    Displays the metadata of a DataFrame
    """

    print(f"{len(df.columns)} columns x {df.count()} rows")
    display(df)

# COMMAND ----------

show_metadata(df_flights_weather_3m)

# COMMAND ----------

import pandas as pd

def find_empty_values_per_column (df: DataFrame) -> None:
    """
    Finds the number of empty values in every column
    WARNING: Takes a long time to complete ~20mins for 3 months
    """
    total = df.count()

    for column in df.columns:
        num_missing = df.filter(f'{column} IS NULL').count()
        percent = f"({(num_missing / total) * 100:.2f}%)" if num_missing > 0 else ""
        print(f"{column}: {num_missing} {percent}")

def find_empty_values_per_column2 (df: DataFrame) -> None:
    """
    Finds the number of empty values in every column
    """
    total = df.count()

    def prepare_nulls (row):
        return [1 if row[column] is None else 0 for column in row.asDict().keys()]

    null_counts = df.rdd.map(prepare_nulls).reduce(lambda a, b: [x + y for x, y in zip(a, b)])

    for i, column in enumerate(df.columns):
        num_missing = null_counts[i]
        percent = f"({(num_missing / total) * 100:.2f}%)" if num_missing > 0 else ""
        print(f"{column}: {num_missing} {percent}")

# COMMAND ----------

find_empty_values_per_column2(df_flights_weather_3m)

# COMMAND ----------

import matplotlib.pyplot as plt
import re

def pyspark_average_by_column (df: DataFrame, column: str, aggregate_column='DEP_DELAY_NEW') -> pd.DataFrame:
    """
    Filters out any rows that contain empty values for the specified column
    and then calculates the average for each unique value in the specified column

    Produces a pandas dataframe (NOT pyspark dataframe)
    """

    sql = f'{column} IS NOT NULL AND CANCELLED = 0 AND DIVERTED = 0'
    return df.filter(sql).groupBy(column).agg({aggregate_column: 'avg'}).toPandas()

def pandas_sort_column (pdf: pd.DataFrame, column: str, map_lambda=None) -> pd.DataFrame:
    """
    Sorts a DataFrame by a column and makes sure it is numeric first

    The map parameter is an optional lambda and can be used to map the column values to a different set of values first
    """

    map_func = lambda x: pd.to_numeric(re.sub('[^0-9\.]','', x if map_lambda is None else map_lambda(x)))
    pdf[column] = pdf[column].apply(map_func)
    return pdf.sort_values(column)

def pandas_moving_average (pdf: pd.DataFrame, column: str, window=10) -> pd.DataFrame:
    """
    Calculates the moving average for a column in a pandas dataframe
    """

    pdf[column] = pdf[column].rolling(window).mean()
    return pdf

def avg_delay_by_day_of_month (df: DataFrame) -> None:
    """
    Plot the average delay for each day of month
    """
    pdf = pyspark_average_by_column(df, 'DAY_OF_MONTH')
    pdf = pandas_sort_column(pdf, 'DAY_OF_MONTH')

    pdf.plot(
        x='DAY_OF_MONTH',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Day of Month',
        legend=False,
        figsize=(12, 6),
        xlabel='Day of Month',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_day_of_month(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_day_of_week (df: DataFrame) -> None:
    """
    Plot the average delay for each day of week
    """
    pdf = pyspark_average_by_column(df, 'DAY_OF_WEEK')
    pdf = pandas_sort_column(pdf, 'DAY_OF_WEEK')

    WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pdf['DAY_OF_WEEK'] = pdf['DAY_OF_WEEK'].apply(lambda x: WEEKDAYS[x - 1])

    pdf.plot(
        x='DAY_OF_WEEK',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Day of Week',
        legend=False,
        figsize=(10, 6),
        xlabel='Day of Week',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_day_of_week(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_month (df: DataFrame) -> None:
    """
    Plot the average delay for each month
    """
    pdf = pyspark_average_by_column(df, 'MONTH')
    pdf = pandas_sort_column(pdf, 'MONTH')

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pdf['MONTH'] = pdf['MONTH'].apply(lambda x: MONTHS[x - 1])

    pdf.plot(
        x='MONTH',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Month',
        legend=False,
        figsize=(10, 6),
        xlabel='Month',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_month(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_time_of_day_dep (df: DataFrame) -> None:
    """
    Plot the average delay for each scheduled departure time of day
    """
    pdf = pyspark_average_by_column(df, 'CRS_DEP_TIME')
    pdf = pandas_sort_column(pdf, 'CRS_DEP_TIME')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=60)

    pdf.plot(
        x='CRS_DEP_TIME',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Scheduled Departure Time of Day',
        legend=False,
        figsize=(10, 6),
        xlabel='Time of Day (24h time)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_time_of_day_dep(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_time_of_day_arr (df: DataFrame) -> None:
    """
    Plot the average delay for each scheduled arrival time of day
    """
    pdf = pyspark_average_by_column(df, 'CRS_ARR_TIME')
    pdf = pandas_sort_column(pdf, 'CRS_ARR_TIME')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=60)

    pdf.plot(
        x='CRS_ARR_TIME',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Scheduled Arrival Time of Day',
        legend=False,
        figsize=(10, 6),
        xlabel='Time of Day (24h time)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_time_of_day_arr(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_distance (df: DataFrame) -> None:
    """
    Plot the average delay by distance
    """
    pdf = pyspark_average_by_column(df, 'DISTANCE')
    pdf = pandas_sort_column(pdf, 'DISTANCE')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=100)

    pdf.plot(
        x='DISTANCE',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Distance',
        legend=False,
        figsize=(10, 6),
        xlabel='Distance (miles)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_distance(df_flights_weather_3m)

# COMMAND ----------

import re

def avg_delay_by_temperature (df: DataFrame) -> None:
    """
    Plot the average delay by distance
    """
    pdf = pyspark_average_by_column(df, 'HourlyDryBulbTemperature')
    pdf = pandas_sort_column(pdf, 'HourlyDryBulbTemperature')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=10)

    pdf.plot(
        x='HourlyDryBulbTemperature',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Temperature',
        legend=False,
        figsize=(10, 6),
        xlabel='Temperature (F)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_temperature(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_altimeter (df: DataFrame) -> None:
    """
    Plot the average delay by altimeter
    """
    pdf = pyspark_average_by_column(df, 'HourlyAltimeterSetting')
    pdf = pandas_sort_column(pdf, 'HourlyAltimeterSetting')

    pdf.plot(
        x='HourlyAltimeterSetting',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Altitude',
        legend=False,
        figsize=(10, 6),
        xlabel='Altitude (in HG)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_altimeter(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_precipitation (df: DataFrame) -> None:
    """
    Plot the average delay by preciptation
    """
    def cleaning_lambda (x):
        if x == 'T':
            return '0.005'
        else:
            return '.'.join(x.split('.')[0:(1 if len(x.split('.')) == 1 else 2)])

    pdf = pyspark_average_by_column(df, 'HourlyPrecipitation')
    pdf = pandas_sort_column(pdf, 'HourlyPrecipitation', map_lambda=cleaning_lambda)
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=2)

    pdf.plot(
        x='HourlyPrecipitation',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Precipitation',
        legend=False,
        figsize=(10, 6),
        xlabel='Precipitation (in/hr)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_precipitation(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_humidity (df: DataFrame) -> None:
    """
    Plot the average delay by humidity
    """
    pdf = pyspark_average_by_column(df, 'HourlyRelativeHumidity')
    pdf = pandas_sort_column(pdf, 'HourlyRelativeHumidity')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=2)

    pdf.plot(
        x='HourlyRelativeHumidity',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Humidity',
        legend=False,
        figsize=(10, 6),
        xlabel='Humidity (%)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_humidity(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_pressure (df: DataFrame) -> None:
    """
    Plot the average delay by pressure
    """
    pdf = pyspark_average_by_column(df, 'HourlyStationPressure')
    pdf = pandas_sort_column(pdf, 'HourlyStationPressure')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=5)

    pdf.plot(
        x='HourlyStationPressure',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Pressure',
        legend=False,
        figsize=(10, 6),
        xlabel='Pressure',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_pressure(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_visibility (df: DataFrame) -> None:
    """
    Plot the average delay by visibility
    """
    pdf = pyspark_average_by_column(df, 'HourlyVisibility')
    pdf = pandas_sort_column(pdf, 'HourlyVisibility')

    pdf.plot(
        x='HourlyVisibility',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Visibility',
        legend=False,
        figsize=(10, 6),
        xlabel='Visibility (100ft)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_visibility(df_flights_weather_3m)

# COMMAND ----------

def avg_delay_by_wind_speed (df: DataFrame) -> None:
    """
    Plot the average delay by wind speed
    """
    pdf = pyspark_average_by_column(df, 'HourlyWindSpeed')
    pdf = pandas_sort_column(pdf, 'HourlyWindSpeed')

    pdf.plot(
        x='HourlyWindSpeed',
        y='avg(DEP_DELAY_NEW)',
        title='Average Delay by Wind Speed',
        legend=False,
        figsize=(10, 6),
        xlabel='Wind Speed (m/h)',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_wind_speed(df_flights_weather_3m)

# COMMAND ----------

def most_delayed_airlines (df: DataFrame) -> None:
    """
    Plot the top 10 most delayed airlines
    """
    pdf = pyspark_average_by_column(df, 'OP_UNIQUE_CARRIER')
    pdf = pdf.sort_values('avg(DEP_DELAY_NEW)')
    pdf = pdf.tail(10)

    pdf.plot(
        x='OP_UNIQUE_CARRIER',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Carrier',
        legend=False,
        figsize=(10, 6),
        xlabel='Carrier',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

most_delayed_airlines(df_flights_weather_3m)

# COMMAND ----------

def most_delayed_departing_airports (df: DataFrame) -> None:
    """
    Plot the top 10 most delayed departing airports
    """
    pdf = pyspark_average_by_column(df, 'ORIGIN')
    pdf = pdf.sort_values('avg(DEP_DELAY_NEW)')
    pdf = pdf.tail(10)

    pdf.plot(
        x='ORIGIN',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Departing Airport',
        legend=False,
        figsize=(10, 6),
        xlabel='Departing Airport',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

most_delayed_departing_airports(df_flights_weather_3m)

# COMMAND ----------

def most_delayed_arriving_airports (df: DataFrame) -> None:
    """
    Plot the top 10 most delayed arriving airports
    """
    pdf = pyspark_average_by_column(df, 'DEST')
    pdf = pdf.sort_values('avg(DEP_DELAY_NEW)')
    pdf = pdf.tail(10)

    pdf.plot(
        x='DEST',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Arriving Airport',
        legend=False,
        figsize=(10, 6),
        xlabel='Arriving Airport',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

most_delayed_arriving_airports(df_flights_weather_3m)

# COMMAND ----------

# Import shapefile for geopandas
!rm -rdf eda_data
!mkdir -p eda_data
!curl https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip -o eda_data/us_shapefiles.zip
!unzip eda_data/us_shapefiles.zip -d eda_data/us_shapefiles
!rm eda_data/us_shapefiles.zip

# COMMAND ----------

import geopandas as gpd
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
    point_gdf.plot(ax=ax, cmap='Oranges_r', marker='o', markersize=8)

origin_airpoint_locations = df_flights_weather_3m \
    .dropDuplicates(['ORIGIN_AIRPORT_ID']) \
    .selectExpr('origin_airport_lat as lat', 'origin_airport_lon as lon') \
    .toPandas()
plot_usa_points(origin_airpoint_locations, title='Average Delay by Origin Airport Locations')

# COMMAND ----------

from functools import cache

@cache
def split_flights_by_canceled (df: DataFrame) -> None:
    """
    Splits the flights dataframe into 3 dataframes:
    - df_flights_canceled: Canceled flights
    - df_flights_diverted: Diverted flights
    - df_flights_success: Successful flights

    This function is memoized so it saves the results for future calls of the same parameter
    """

    df_flights_canceled = df.filter('CANCELLED == 1')
    df_flights_diverted = df.filter('DIVERTED == 1')
    df_flights_success = df.filter('CANCELLED == 0 AND DIVERTED == 0')

    return df_flights_canceled, df_flights_diverted, df_flights_success

def canc_divert_by_day_of_month (df: DataFrame) -> None:
    """
    Plot the number of canceled and diverted flights by day of month
    """
    df_flights_canc, df_flights_divert, df_flights_success = split_flights_by_canceled(df)

    pdf_canceled = df_flights_canc.groupBy('DAY_OF_MONTH').count().toPandas()
    pdf_diverted = df_flights_divert.groupBy('DAY_OF_MONTH').count().toPandas()
    pdf_success = df_flights_success.groupBy('DAY_OF_MONTH').count().toPandas()

    pdf_canceled = pandas_sort_column(pdf_canceled, 'DAY_OF_MONTH')
    pdf_diverted = pandas_sort_column(pdf_diverted, 'DAY_OF_MONTH')
    pdf_success = pandas_sort_column(pdf_success, 'DAY_OF_MONTH')

    total_flights = pdf_success['count'] + pdf_diverted['count'] + pdf_canceled['count']
    pdf_canceled['percent_canceled'] = pdf_canceled['count'] / total_flights * 100
    pdf_diverted['percent_diverted'] = pdf_diverted['count'] / total_flights * 100

    pdf_canceled.plot(
        x='DAY_OF_MONTH',
        y='percent_canceled',
        kind='bar',
        title='Canceled Flights by Day of Month',
        legend=False,
        figsize=(12, 6),
        xlabel='Day of Month',
        ylabel='Canceled Flights (%)'
    )

    pdf_diverted.plot(
        x='DAY_OF_MONTH',
        y='percent_diverted',
        kind='bar',
        title='Diverted Flights by Day of Month',
        legend=False,
        figsize=(12, 6),
        xlabel='Day of Month',
        ylabel='Diverted Flights (%)'
    )

canc_divert_by_day_of_month(df_flights_weather_3m)

# COMMAND ----------

def canc_divert_by_day_of_week (df: DataFrame) -> None:
    """
    Plot the number of canceled and diverted flights by day of week
    """
    df_flights_canc, df_flights_divert, df_flights_success = split_flights_by_canceled(df)

    pdf_canceled = df_flights_canc.groupBy('DAY_OF_WEEK').count().toPandas()
    pdf_diverted = df_flights_divert.groupBy('DAY_OF_WEEK').count().toPandas()
    pdf_success = df_flights_success.groupBy('DAY_OF_WEEK').count().toPandas()

    pdf_canceled = pandas_sort_column(pdf_canceled, 'DAY_OF_WEEK')
    pdf_diverted = pandas_sort_column(pdf_diverted, 'DAY_OF_WEEK')
    pdf_success = pandas_sort_column(pdf_success, 'DAY_OF_WEEK')

    total_flights = pdf_success['count'] + pdf_diverted['count'] + pdf_canceled['count']
    pdf_canceled['percent_canceled'] = pdf_canceled['count'] / total_flights * 100
    pdf_diverted['percent_diverted'] = pdf_diverted['count'] / total_flights * 100

    WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pdf_canceled['DAY_OF_WEEK'] = pdf_canceled['DAY_OF_WEEK'].apply(lambda x: WEEKDAYS[x - 1])
    pdf_diverted['DAY_OF_WEEK'] = pdf_diverted['DAY_OF_WEEK'].apply(lambda x: WEEKDAYS[x - 1])

    pdf_canceled.plot(
        x='DAY_OF_WEEK',
        y='percent_canceled',
        kind='bar',
        title='Canceled Flights by Day of Week',
        legend=False,
        figsize=(12, 6),
        xlabel='Day of Week',
        ylabel='Canceled Flights (%)'
    )

    pdf_diverted.plot(
        x='DAY_OF_WEEK',
        y='percent_diverted',
        kind='bar',
        title='Diverted Flights by Day of Week',
        legend=False,
        figsize=(12, 6),
        xlabel='Day of Week',
        ylabel='Diverted Flights (%)'
    )

canc_divert_by_day_of_week(df_flights_weather_3m)

# COMMAND ----------

def canc_divert_by_month (df: DataFrame) -> None:
    """
    Plot the number of canceled and diverted flights by month
    """
    df_flights_canc, df_flights_divert, df_flights_success = split_flights_by_canceled(df)

    pdf_canceled = df_flights_canc.groupBy('MONTH').count().toPandas()
    pdf_diverted = df_flights_divert.groupBy('MONTH').count().toPandas()
    pdf_success = df_flights_success.groupBy('MONTH').count().toPandas()

    pdf_canceled = pandas_sort_column(pdf_canceled, 'MONTH')
    pdf_diverted = pandas_sort_column(pdf_diverted, 'MONTH')
    pdf_success = pandas_sort_column(pdf_success, 'MONTH')

    total_flights = pdf_success['count'] + pdf_diverted['count'] + pdf_canceled['count']    
    pdf_canceled['percent_canceled'] = pdf_canceled['count'] / total_flights * 100
    pdf_diverted['percent_diverted'] = pdf_diverted['count'] / total_flights * 100

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pdf_canceled['MONTH'] = pdf_canceled['MONTH'].apply(lambda x: MONTHS[x - 1])
    pdf_diverted['MONTH'] = pdf_diverted['MONTH'].apply(lambda x: MONTHS[x - 1])

    pdf_canceled.plot(
        x='MONTH',
        y='percent_canceled',
        kind='bar',
        title='Canceled Flights by Month',
        legend=False,
        figsize=(12, 6),
        xlabel='Month',
        ylabel='Canceled Flights (%)'
    )

    pdf_diverted.plot(
        x='MONTH',
        y='percent_diverted',
        kind='bar',
        title='Diverted Flights by Month',
        legend=False,
        figsize=(12, 6),
        xlabel='Month',
        ylabel='Diverted Flights (%)'
    )

canc_divert_by_month(df_flights_weather_3m)

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO: make boxplots show full 5 figure summary instead of just count

# COMMAND ----------

def canc_divert_by_scheduled_departure (df: DataFrame) -> None:
    """
    Plot the number of canceled and diverted flights by scheduled depature time
    """
    df_flights_canc, df_flights_divert, df_flights_success = split_flights_by_canceled(df)

    def round_time (x):
        """
        Floors the time to the hour
        """
        for pdf in x:
            pdf['CRS_DEP_TIME'] = pdf['CRS_DEP_TIME'].apply(lambda x: x[:-2])
            yield pdf

    pdf_canceled = df_flights_canc.mapInPandas(round_time, df_flights_canc.schema).groupBy('CRS_DEP_TIME').count().toPandas()
    pdf_diverted = df_flights_divert.mapInPandas(round_time, df_flights_canc.schema).groupBy('CRS_DEP_TIME').count().toPandas()
    pdf_success = df_flights_success.mapInPandas(round_time, df_flights_canc.schema).groupBy('CRS_DEP_TIME').count().toPandas()

    pdf_canceled = pandas_sort_column(pdf_canceled, 'CRS_DEP_TIME')
    pdf_diverted = pandas_sort_column(pdf_diverted, 'CRS_DEP_TIME')
    pdf_success = pandas_sort_column(pdf_success, 'CRS_DEP_TIME')

    total_flights = pdf_success['count'] + pdf_diverted['count'] + pdf_canceled['count'] 
    pdf_canceled['percent_canceled'] = pdf_canceled['count'] / total_flights * 100
    pdf_diverted['percent_diverted'] = pdf_diverted['count'] / total_flights * 100

    pdf_canceled.boxplot(
        column=['percent_canceled'],
        by='CRS_DEP_TIME',
        figsize=(12, 6)
    )

    pdf_diverted.boxplot(
        column=['percent_diverted'],
        by='CRS_DEP_TIME',
        figsize=(12, 6)
    )

canc_divert_by_scheduled_departure(df_flights_weather_3m)

# COMMAND ----------

def canc_divert_by_scheduled_arrival (df: DataFrame) -> None:
    """
    Plot the number of canceled and diverted flights by scheduled arrival time
    """
    df_flights_canc, df_flights_divert, df_flights_success = split_flights_by_canceled(df)

    pdf_canceled = df_flights_canc.groupBy('CRS_ARR_TIME').count().toPandas()
    pdf_diverted = df_flights_divert.groupBy('CRS_ARR_TIME').count().toPandas()
    pdf_success = df_flights_success.groupBy('CRS_ARR_TIME').count().toPandas()

    pdf_canceled = pandas_sort_column(pdf_canceled, 'CRS_ARR_TIME')
    pdf_diverted = pandas_sort_column(pdf_diverted, 'CRS_ARR_TIME')
    pdf_success = pandas_sort_column(pdf_success, 'CRS_ARR_TIME')

    total_flights = pdf_success['count'] + pdf_diverted['count'] + pdf_canceled['count'] 
    pdf_canceled['percent_canceled'] = pdf_canceled['count'] / total_flights * 100
    pdf_diverted['percent_diverted'] = pdf_diverted['count'] / total_flights * 100

    pdf_canceled['CRS_ARR_TIME'] = pdf_canceled['CRS_ARR_TIME'].apply(lambda x: round(x / 100))
    pdf_diverted['CRS_ARR_TIME'] = pdf_diverted['CRS_ARR_TIME'].apply(lambda x: round(x / 100))

    pdf_canceled.boxplot(
        column=['percent_canceled'],
        by='CRS_ARR_TIME',
        figsize=(12, 6)
    )

    pdf_diverted.boxplot(
        column=['percent_diverted'],
        by='CRS_ARR_TIME',
        figsize=(12, 6)
    )

canc_divert_by_scheduled_arrival(df_flights_weather_3m)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1Y EDA

# COMMAND ----------

df_flights_weather_1y = get_data_from_cloud('OTPW_12M/OTPW_12M_2015.csv.gz')

# COMMAND ----------

show_metadata(df_flights_weather_1y)

# COMMAND ----------

find_empty_values_per_column2(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_day_of_month(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_day_of_week(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_month(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_time_of_day_dep(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_time_of_day_arr(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_distance(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_temperature(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_altimeter(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_precipitation(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_humidity(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_pressure(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_visibility(df_flights_weather_1y)

# COMMAND ----------

avg_delay_by_wind_speed(df_flights_weather_1y)

# COMMAND ----------

most_delayed_airlines(df_flights_weather_1y)

# COMMAND ----------

most_delayed_departing_airports(df_flights_weather_1y)

# COMMAND ----------

most_delayed_arriving_airports(df_flights_weather_1y)