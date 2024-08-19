# Databricks notebook source
DATA_BASE_DIR = "dbfs:/mnt/mids-w261/"
# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
blob_container  = "ds261-final-project-team-2-2"       # The name of your container created in https://portal.azure.com
storage_account = "ds261team"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "261_team_2_2_scope"           # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "261_team_2_2_key"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

def connect_cloud_database () -> None:
    """
    Connects this notebook to the cloud database
    """

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

def get_data_from_cloud (dataset: Literal['OTPW_3M_2015.csv', 'OTPW_12M/OTPW_12M_2015.csv.gz', 'OTPW_60M']) -> DataFrame:
    """
    Gets the dataset from the cloud
    """

    if dataset == 'OTPW_60M':
        df_flights_weather_5y = spark.read.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_60M")
        return df_flights_weather_5y.filter("YEAR = 2015 OR YEAR = 2016 OR YEAR = 2017 OR YEAR = 2018").cache()

    return spark.read.format("csv").option("header","true").load(f"{DATA_BASE_DIR}{dataset}")

df_flights_weather_4y = get_data_from_cloud('OTPW_60M')

# COMMAND ----------

def show_metadata (df: DataFrame) -> None:
    """
    Displays the metadata of a DataFrame
    """

    print(f"{len(df.columns)} columns x {df.count()} rows")
    display(df)

# COMMAND ----------

show_metadata(df_flights_weather_4y)

# COMMAND ----------

print(df_flights_weather_4y.select('DEP_DELAY_NEW').summary().show())

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

find_empty_values_per_column2(df_flights_weather_4y)

# COMMAND ----------

import matplotlib.pyplot as plt
import re

def pyspark_average_by_column (df: DataFrame, column: str, aggregate_column='DEP_DELAY_NEW', filter_cancel=True) -> pd.DataFrame:
    """
    Filters out any rows that contain empty values for the specified column
    and then calculates the average for each unique value in the specified column

    Produces a pandas dataframe (NOT pyspark dataframe)
    """

    sql = f'{column} IS NOT NULL AND CANCELLED = 0 AND DIVERTED = 0' if filter_cancel else f'{column} IS NOT NULL'
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

avg_delay_by_day_of_month(df_flights_weather_4y)

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

avg_delay_by_day_of_week(df_flights_weather_4y)

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

avg_delay_by_month(df_flights_weather_4y)

# COMMAND ----------

def avg_delay_by_year (df: DataFrame) -> None:
    """
    Plot the average delay for each year
    """
    pdf = pyspark_average_by_column(df, 'YEAR')
    pdf = pandas_sort_column(pdf, 'YEAR')

    pdf.plot(
        x='YEAR',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Year',
        legend=False,
        figsize=(10, 6),
        xlabel='Year',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_year(df_flights_weather_4y)

# COMMAND ----------

from pyspark.sql.types import IntegerType

def avg_delay_by_date (df: DataFrame) -> None:
    """
    Plot the average delay for each date
    """
    ndf = df.withColumn('date_index', df["YEAR"].cast(IntegerType()) * 10000 + df["MONTH"].cast(IntegerType()) * 100 + df["DAY_OF_MONTH"].cast(IntegerType()))
    pdf = pyspark_average_by_column(ndf, 'date_index')
    pdf = pdf.sort_values('date_index')
    pdf['FL_DATE'] = pdf['date_index'].apply(lambda x: str(x // 10000) + '-' + str(x // 100 % 100) + '-' + str(x % 100))
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=20)

    pdf.plot(
        x='FL_DATE',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Date',
        legend=False,
        figsize=(10, 6),
        xlabel='Date',
        ylabel='Average Delay (mins)'
    )

# COMMAND ----------

avg_delay_by_date(df_flights_weather_4y)

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

avg_delay_by_time_of_day_dep(df_flights_weather_4y)

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

avg_delay_by_time_of_day_arr(df_flights_weather_4y)

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

avg_delay_by_distance(df_flights_weather_4y)

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

avg_delay_by_temperature(df_flights_weather_4y)

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

avg_delay_by_altimeter(df_flights_weather_4y)

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

avg_delay_by_precipitation(df_flights_weather_4y)

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

avg_delay_by_humidity(df_flights_weather_4y)

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

avg_delay_by_pressure(df_flights_weather_4y)

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

avg_delay_by_visibility(df_flights_weather_4y)

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

avg_delay_by_wind_speed(df_flights_weather_4y)

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

most_delayed_airlines(df_flights_weather_4y)

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

most_delayed_departing_airports(df_flights_weather_4y)

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

most_delayed_arriving_airports(df_flights_weather_4y)

# COMMAND ----------

def get_cleaned_data():
    parquet_file_path = f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y"

    # Read the Parquet files into a Spark DataFrame
    return spark.read.parquet(parquet_file_path).cache()

clean_4y = get_cleaned_data()

# COMMAND ----------

show_metadata(clean_4y)

# COMMAND ----------

def eda_is_vert_obscured(df):
    obscured_df = df.filter('is_vert_obscured = true')
    unobscured_df = df.filter('is_vert_obscured = false')

    obscured_df.select('DEP_DELAY_NEW').summary().show()
    unobscured_df.select('DEP_DELAY_NEW').summary().show()

eda_is_vert_obscured(clean_4y)

# COMMAND ----------

def avg_delay_by_time_bucket (df: DataFrame) -> None:
    """
    Plot the average delay for each departure time bucket
    """
    pdf = pyspark_average_by_column(df, 'Departure_Time_Buckets', filter_cancel=False)

    buckets = ['Morning', 'Afternoon', 'Evening', 'Night']
    pdf['bucket_index'] = pdf['Departure_Time_Buckets'].apply(lambda x: buckets.index(x))
    pdf = pdf.sort_values('bucket_index')

    pdf.plot(
        x='Departure_Time_Buckets',
        y='avg(DEP_DELAY_NEW)',
        kind='bar',
        title='Average Delay by Departure Time Bucket',
        legend=False,
        figsize=(10, 6),
        xlabel='Departure Time Bucket',
        ylabel='Average Delay (mins)'
    )

avg_delay_by_time_bucket(clean_4y)

# COMMAND ----------

def avg_delay_by_prev_elapsed_time (df: DataFrame) -> None:
    """
    Plot the average delay by the elapsed time from the previous flight
    """
    pdf = pyspark_average_by_column(df, 'PrevFlightElapsedTime', filter_cancel=False)
    pdf = pdf.sort_values('PrevFlightElapsedTime')
    pdf = pandas_moving_average(pdf, 'avg(DEP_DELAY_NEW)', window=20)

    pdf.plot(
        x='PrevFlightElapsedTime',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Previous Flight Elapsed Time',
        legend=False,
        figsize=(10, 6),
        xlabel='Previous Flight Elapsed Time (mins)',
        ylabel='Average Delay (mins)'
    )

avg_delay_by_prev_elapsed_time(clean_4y)

# COMMAND ----------

def prev_flight_delay_correlation(df, downsampled_df):
    """
    Print the pearson correlation coefficient between the previous flight delay and the actual delay
    """

    print(df.corr('DEP_DELAY_NEW', 'PrevFlightDelay'))
    pdf = downsampled_df.toPandas()

    pdf.plot(
        x='PrevFlightDelay',
        y='DEP_DELAY_NEW',
        title='Delay vs Previous Flight Delay (Downsampled)',
        kind='scatter',
        legend=False,
        figsize=(10, 6),
        xlabel='Previous Flight Delay (mins)',
        ylabel='Flight Delay (mins)'
    )

    pdf2 = pyspark_average_by_column(df, 'PrevFlightDelay', filter_cancel=False)
    pdf2 = pdf2.sort_values('PrevFlightDelay')
    pdf2 = pandas_moving_average(pdf2, 'avg(DEP_DELAY_NEW)', window=20)

    pdf2.plot(
        x='PrevFlightDelay',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Previous Flight Delay',
        legend=False,
        figsize=(10, 6),
        xlabel='Previous Flight Delay (mins)',
        ylabel='Average Delay (mins)'
    )

clean_4y_downsampled, _ = clean_4y.randomSplit([0.01, 0.99], 2024)
prev_flight_delay_correlation(clean_4y, clean_4y_downsampled)

# COMMAND ----------

def eda_is_extreme_weather(df):
    obscured_df = df.filter('is_extreme_weather = true')
    unobscured_df = df.filter('is_extreme_weather = false')

    obscured_df.select('DEP_DELAY_NEW').summary().show()
    unobscured_df.select('DEP_DELAY_NEW').summary().show()

eda_is_extreme_weather(clean_4y)

# COMMAND ----------

def eda_is_holiday(df):
    obscured_df = df.filter('is_holiday = true')
    unobscured_df = df.filter('is_holiday = false')

    obscured_df.select('DEP_DELAY_NEW').summary().show()
    unobscured_df.select('DEP_DELAY_NEW').summary().show()

eda_is_holiday(clean_4y)

# COMMAND ----------

def prev_flight_origin_pagerank(df, downsampled_df):
    """
    Print the pearson correlation coefficient between the origin pagerank and the actual delay
    """

    print(df.corr('DEP_DELAY_NEW', 'origin_pagerank'))
    pdf = downsampled_df.toPandas()

    pdf.plot(
        x='origin_pagerank',
        y='DEP_DELAY_NEW',
        title='Delay vs Origin Pagerank (Downsampled)',
        kind='scatter',
        legend=False,
        figsize=(10, 6),
        xlabel='Origin Pagerank ',
        ylabel='Flight Delay (mins)'
    )

    pdf2 = pyspark_average_by_column(df, 'origin_pagerank', filter_cancel=False)
    pdf2 = pdf2.sort_values('origin_pagerank')
    pdf2 = pandas_moving_average(pdf2, 'avg(DEP_DELAY_NEW)', window=10)

    pdf2.plot(
        x='origin_pagerank',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Origin Pagerank',
        legend=False,
        figsize=(10, 6),
        xlabel='Origin Pagerank',
        ylabel='Average Delay (mins)'
    )

prev_flight_origin_pagerank(clean_4y, clean_4y_downsampled)

# COMMAND ----------

def prev_flight_dest_pagerank(df, downsampled_df):
    """
    Print the pearson correlation coefficient between the dest pagerank and the actual delay
    """

    print(df.corr('DEP_DELAY_NEW', 'dest_pagerank'))
    pdf = downsampled_df.toPandas()

    pdf.plot(
        x='dest_pagerank',
        y='DEP_DELAY_NEW',
        title='Delay vs Destination Pagerank (Downsampled)',
        kind='scatter',
        legend=False,
        figsize=(10, 6),
        xlabel='Destination Pagerank ',
        ylabel='Flight Delay (mins)'
    )

    pdf2 = pyspark_average_by_column(df, 'dest_pagerank', filter_cancel=False)
    pdf2 = pdf2.sort_values('dest_pagerank')
    pdf2 = pandas_moving_average(pdf2, 'avg(DEP_DELAY_NEW)', window=10)

    pdf2.plot(
        x='dest_pagerank',
        y='avg(DEP_DELAY_NEW)',
        title='Smoothed Average Delay by Destination Pagerank',
        legend=False,
        figsize=(10, 6),
        xlabel='Destination Pagerank',
        ylabel='Average Delay (mins)'
    )

prev_flight_dest_pagerank(clean_4y, clean_4y_downsampled)