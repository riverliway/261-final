# Databricks notebook source
# MAGIC %md
# MAGIC # Connecting to data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connects this notebook to the cloud database

# COMMAND ----------

DATA_BASE_DIR = "dbfs:/mnt/mids-w261/"

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

# COMMAND ----------

# list blob storage root folder 
display(dbutils.fs.ls(f"{DATA_BASE_DIR}"))

# COMMAND ----------

# list OTPW_60M folder 
display(dbutils.fs.ls(f"{DATA_BASE_DIR}/OTPW_60M"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load 3-month, 1-year and 5-year CSVs into DataFrames

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
from typing import Literal

def get_data_from_cloud (dataset_filename: str) -> DataFrame:
    """
    Gets the dataset from the cloud
    """

    return spark.read.format("csv").option("header","true").load(f"{DATA_BASE_DIR}{dataset_filename}")
# not needed since we read dfs from parquet (a few cells below)
# df_flights_weather_3m = get_data_from_cloud('OTPW_3M_2015.csv')
# df_flights_weather_1y = get_data_from_cloud('OTPW_12M/OTPW_12M_2015.csv.gz')    # 1y

# df_flights_weather_5y = spark.read.option("header",True) \
#                        .option("compression", "gzip") \
#                        .option("delimiter",",") \
#                        .csv([f"{DATA_BASE_DIR}OTPW_60M"])

# COMMAND ----------

# write data from CSV to blob storage as parquet format for performance
# only needed once
# df_flights_weather_3m.write.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_3M_2015", mode="overwrite")
# df_flights_weather_1y.write.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_12M_2015", mode="overwrite")
# df_flights_weather_5y.write.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_60M", mode="overwrite")

# display(dbutils.fs.ls(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_60M"))

# COMMAND ----------

# Repartition data (for performance) + cache DFs
# not needed if reading from parquet
# df_flights_weather_3m = df_flights_weather_3m.repartition(10).cache()
# df_flights_weather_1y = df_flights_weather_1y.repartition(200).cache()
# df_flights_weather_5y = df_flights_weather_5y.repartition(300).cache()

# COMMAND ----------

# read dfs from parquet
df_flights_weather_3m = spark.read.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_3M_2015").cache()
df_flights_weather_1y = spark.read.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_12M_2015").cache()
df_flights_weather_5y = spark.read.parquet(f"{team_blob_url}/Phase_3/OTPW_parquet/OTPW_60M")

df_flights_weather_5y = df_flights_weather_5y.repartition(300).cache()

# COMMAND ----------

def show_metadata (df: DataFrame) -> None:
    """
    Displays the metadata of a DataFrame
    """

    print(f"{len(df.columns)} columns x {df.count()} rows")
    display(df)

# COMMAND ----------

# show_metadata(df_flights_weather_3m)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, DoubleType, DateType, FloatType, DoubleType, BooleanType
from pyspark.sql.functions import to_timestamp, lpad, concat_ws, when, col, date_add, hour, lag, unix_timestamp, to_date, lit
from pyspark.sql import Window
import holidays


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Grabbing relevant columns and rows

# COMMAND ----------

def filter_relevant_flights(df):
    """
    Filter out diverted and cancelled flights    
    """
    df = df.filter((df.CANCELLED == 0) & (df.DIVERTED == 0))
    return df

# COMMAND ----------

def drop_duplicates_records(df):
    """ drop duplicates of flight records"""
    return df.dropDuplicates(['FL_DATE', 'TAIL_NUM', 'OP_CARRIER', 'DEP_TIME'])

# COMMAND ----------


def select_columns(df):
    """
    Selects relevant columns and corrects data types
    """
    columns = [
        "YEAR", # for dataset spliting
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "MONTH",
        "ORIGIN",
        "DEST",
        "TAIL_NUM",
        "FL_DATE",    
        "OP_CARRIER",
        "CRS_DEP_TIME",
        "DEP_DELAY_NEW",
        "CRS_ARR_TIME",
        "ARR_DELAY_NEW",
        "ACTUAL_ELAPSED_TIME",
        #"CANCELLED",
        #"DIVERTED",
        "DISTANCE",
        "ELEVATION",
        #"CARRIER_DELAY",
        #"WEATHER_DELAY",
        #"NAS_DELAY",
        #"SECURITY_DELAY",
        #"LATE_AIRCRAFT_DELAY",
        "HourlyDryBulbTemperature",
        "HourlyPrecipitation",
        "HourlyRelativeHumidity",
        "HourlyStationPressure",
        "HourlyVisibility",
        "HourlyWindSpeed",
        "HourlySkyConditions",
        "HourlyPresentWeatherType"
    ]
    data_types = {
        "YEAR": StringType(),
        "DAY_OF_MONTH": IntegerType(),
        "DAY_OF_WEEK": StringType(),
        "MONTH": IntegerType(),
        "FL_DATE": DateType(),
        "OP_CARRIER": StringType(),
        "ORIGIN": StringType(),
        "DEST": StringType(),
        "TAIL_NUM": StringType(),
        "CRS_DEP_TIME": StringType(),
        "DEP_DELAY_NEW": DoubleType(),
        "CRS_ARR_TIME": StringType(),
        "ARR_DELAY_NEW": DoubleType(),
        "ACTUAL_ELAPSED_TIME": DoubleType(),
        #"CANCELLED": IntegerType(),
        #"DIVERTED": IntegerType(),
        "DISTANCE": DoubleType(),
        "ELEVATION": DoubleType(),
        #"CARRIER_DELAY": DoubleType(),
        #"WEATHER_DELAY": DoubleType(),
        #"NAS_DELAY": DoubleType(),
        #"SECURITY_DELAY": DoubleType(),
        #"LATE_AIRCRAFT_DELAY": DoubleType(),
        "HourlySkyConditions": StringType(),
        "HourlyDryBulbTemperature": DoubleType(),
        "HourlyPrecipitation": DoubleType(),
        "HourlyRelativeHumidity": DoubleType(),
        "HourlyStationPressure": DoubleType(),
        "HourlyVisibility": DoubleType(),
        "HourlyWindSpeed": DoubleType(),
        "HourlyPresentWeatherType": StringType()
    }

    # Apply the data types to each column
    for column, dtype in data_types.items():
        df = df.withColumn(column, col(column).cast(dtype))
    df_clean = df.select(columns)

    # Show the schema to verify the changes
    df_clean.printSchema()

    return df_clean

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Deriving New Features 

# COMMAND ----------

def create_vert_obscuration_flag(df):
    """
    Create a new column named 'is_vert_obscured' that contains 1 if the sky condition at any point is obscured, 0 otherwise
    """
    df = df.withColumn("is_vert_obscured", when(col("HourlySkyConditions").contains('VV'), True).otherwise(False)) 
    return df

# COMMAND ----------

def convert_string_to_time(df):
    """
    Convert the CRS_DEP_TIME and CRS_ARR_TIME columns to a timestamp format.
    """
    df = df.withColumn("CRS_DEP_TIME", to_timestamp(concat_ws(' ', col("FL_DATE"), lpad(col("CRS_DEP_TIME"), 4, '0')), 'yyyy-MM-dd HHmm'))
    df = df.withColumn("CRS_ARR_TIME", to_timestamp(concat_ws(' ', col("FL_DATE"), lpad(col("CRS_ARR_TIME"), 4, '0')), 'yyyy-MM-dd HHmm'))
    df = df.withColumn("CRS_ARR_TIME", when(col("CRS_ARR_TIME") < col("CRS_DEP_TIME"), date_add(col("CRS_ARR_TIME"), 1)).otherwise(col("CRS_ARR_TIME")))
    return df

# COMMAND ----------

def create_departure_bins(df):
    """
    Categorizes departure time by time of day
    """
    df = df.withColumn("Departure_Time_Buckets", 
                       when((hour(col("CRS_DEP_TIME")) >= 6) & (hour(col("CRS_DEP_TIME")) < 12), "Morning")
                       .when((hour(col("CRS_DEP_TIME")) >= 12) & (hour(col("CRS_DEP_TIME")) < 18), "Afternoon")
                       .when((hour(col("CRS_DEP_TIME")) >= 18) & (hour(col("CRS_DEP_TIME")) < 24), "Evening")
                       .otherwise("Night"))
    return df

# COMMAND ----------

def generate_num_of_hours_since_last_flight(df):
    """
    Generate a new column that will count the number of hours since the last flight for a particular airline/airport
    Ignore flights where previous flight is less than 2 hours
    """
    
    window_spec = Window.partitionBy("TAIL_NUM").orderBy("CRS_ARR_TIME")
    df = df.withColumn("PrevFlightArrive", lag(col("CRS_ARR_TIME")).over(window_spec))
    df = df.withColumn("PrevFlightElapsedTime", lag(col("ACTUAL_ELAPSED_TIME")).over(window_spec))
    df = df.withColumn("HoursSinceLastFlight", 
                       when(col("PrevFlightElapsedTime") >= 120, 
                            (unix_timestamp(col("CRS_DEP_TIME")) - unix_timestamp(col("PrevFlightArrive"))) / 3600)
                       .otherwise(None))
    df = df.fillna({"HoursSinceLastFlight": 0})
    return df

# COMMAND ----------

def generate_previous_flight_delay_minutes(df):
    """
    Generate a new column that will count the number of hours since the last flight for a particular airline/airport
    Ignore flights where previous flight is less than 2 hours
    """
    
    window_spec = Window.partitionBy("TAIL_NUM").orderBy("CRS_ARR_TIME")
    df = df.withColumn("PrevFlightArrive", lag(col("CRS_ARR_TIME")).over(window_spec))
    df = df.withColumn("PrevFlightDelay", 
                       when(lag(col("DEP_DELAY_NEW")).over(window_spec).isNull(), 0)
                       .otherwise(lag(col("DEP_DELAY_NEW")).over(window_spec)))     
    """
    I had to remove the 2 hour logic because a lot of the flights would get removed
    df = df.withColumn("PrevFlightDelay", 
                       when(col("PrevFlightElapsedTime") >= 120, col("PrevFlightElapsedTime"))
                       .otherwise(None))
    """
    return df

# COMMAND ----------

def count_and_drop_rows_with_na(df):
    """
    Dropping NAs.
    Majority comes from the precipitation field which means data wasn't reported
    """
    initial_count = df.count()
    df_clean = df.dropna()
    
    final_count = df_clean.count()
    dropped_count = initial_count - final_count
    
    return dropped_count, df_clean

    print(dropped_count)

# COMMAND ----------

def generate_weather_boolean(df):
    """
    Parses HourlyPresentWeatherType and flags if weather is extreme.

    We only look for these conditions:
    GR:07 - Hail
    GS:08 - Small Hail and/or Snow Pellets
    VA:4 - Volcanic Ash
    DU:5 - Widespread Dust
    SA:6 - Sand
    PO:1 - Well developed dust/sand whirls
    SQ:2 - Squalls
    FC:3 - Funnel Cloud, Waterspout or Tornado
    SS:4 - Sandstorm
    DS:5 - Duststorm
    """

    extreme_weather_conditions = ["GR", "GS", "VA", "DU", "SA", "PO", "SQ", "FC", "SS", "DS"]
    # UDF to check for extreme weather conditions
    def is_extreme_weather(description):
        if description is None or len(description) == 0:
            return False
        description_upper = description.upper()
        for condition in extreme_weather_conditions:
            if condition in description_upper:
                return True
        return False

    # Register UDF
    is_extreme_weather_udf = udf(is_extreme_weather, BooleanType())

    # Add a column with the boolean flag
    df_with_flag = df.withColumn("is_extreme_weather", is_extreme_weather_udf(col("HourlyPresentWeatherType")))

    return df_with_flag

# COMMAND ----------

def limit_delay_tail(df):
    """
    DEP_DELAY_NEW has a long tail and we want to focus on the middle
    This function will cap the delay by the 95th percentile
    """
    percentile_value = df.approxQuantile("DEP_DELAY_NEW", [0.95], 0.01)[0]
    print(percentile_value)
    # Cap the long tail using the dynamic threshold
    df = df.withColumn("dep_capped_delay", when(col("DEP_DELAY_NEW") > percentile_value, percentile_value).otherwise(col("DEP_DELAY_NEW")))

    return df

# COMMAND ----------

# Global cache to store holidays for each year
holiday_cache = {}

def get_us_holidays(year):
    """
    Retrieve the US holidays for a given year, using a cache to avoid recomputation.
    """
    if year not in holiday_cache:
        holiday_cache[year] = holidays.US(years=year)
    return holiday_cache[year]

def add_holiday_flag(df, date_col):
    """
    Add a holiday flag to the DataFrame based on the date column.
    """

    def is_holiday(date):
        us_holidays = get_us_holidays(date.year)
        return date in us_holidays

    # Register the UDF
    is_holiday_udf = udf(lambda date: is_holiday(date), BooleanType())

    # Convert the date column to DateType if it's not already
    df = df.withColumn(date_col, to_date(col(date_col)))

    # Add the holiday flag column
    df = df.withColumn("is_holiday", is_holiday_udf(col(date_col)))

    return df

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def one_hot_encode(df):
    """
    Convert categorical columns to one-hot encoded columns.
    """
    categorical_columns = ['ORIGIN', 'DEST', 'OP_CARRIER', 'Departure_Time_Buckets', 'DAY_OF_WEEK']

    stages = []
    for col in categorical_columns:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_onehot")
        stages += [indexer, encoder]
    
    pipeline = Pipeline(stages=stages)
    df = pipeline.fit(df).transform(df)
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Page rank feature

# COMMAND ----------

# pagerank
from graphframes import GraphFrame

def add_pagerank_features(df):
  """
  we treat the flight network as a graph. In this graph:
    Airports are nodes (vertices)
    Flights between airports are edges (links)
  The PageRank algorithm will help determine the importance of each airport. This importance can influence flight delays due to factors like congestion, connectivity, or being a major hub.
  """
  # Define the vertices DataFrame
  airports = df.select("ORIGIN").union(df.select("DEST")).distinct().withColumnRenamed("ORIGIN", "id")

  # Define the edges DataFrame
  flights = df.select("ORIGIN", "DEST").withColumnRenamed("ORIGIN", "src").withColumnRenamed("DEST", "dst")

  # Create the GraphFrame
  g = GraphFrame(airports, flights)

  # Run PageRank
  results = g.pageRank(resetProbability=0.15, maxIter=15)

  # Extract the PageRank values
  pagerank_results = results.vertices.select("id", "pagerank")

  # Join the PageRank results back to the flight data
  df = df.join(pagerank_results, df.ORIGIN == pagerank_results.id, "left").drop("id")
  df = df.withColumnRenamed("pagerank", "origin_pagerank")

  df = df.join(pagerank_results, df.DEST == pagerank_results.id, "left").drop("id")
  df = df.withColumnRenamed("pagerank", "dest_pagerank")

  return df

# COMMAND ----------

from graphframes import GraphFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

def add_pagerank_features_blocks(df):
    """
    We treat the flight network as a graph. In this graph:
      Airports are nodes (vertices)
      Flights between airports are edges (links)
    The PageRank algorithm will help determine the importance of each airport. This importance can influence flight delays due to factors like congestion, connectivity, or being a major hub.
    """
    
    # Create an empty DataFrame to collect results
    final_df = None

    # Get the unique years from the DataFrame
    years = df.select("YEAR").distinct().rdd.flatMap(lambda x: x).collect()
    print(f"add_pagerank_features_blocks for years {years}")
    
    for year in years:
        print(f"add_pagerank_features_blocks for year {year}")
        # Filter the DataFrame for the current year
        df_year = df.filter(df.YEAR == year)
        
        # Define the vertices DataFrame
        airports = df_year.select("ORIGIN").union(df_year.select("DEST")).distinct().withColumnRenamed("ORIGIN", "id")

        # Define the edges DataFrame
        flights = df_year.select("ORIGIN", "DEST").withColumnRenamed("ORIGIN", "src").withColumnRenamed("DEST", "dst")

        # Create the GraphFrame
        g = GraphFrame(airports, flights)

        # Run PageRank
        results = g.pageRank(resetProbability=0.15, maxIter=15)

        # Extract the PageRank values
        pagerank_results = results.vertices.select("id", "pagerank")

        # Join the PageRank results back to the flight data
        df_year = df_year.join(pagerank_results, df_year.ORIGIN == pagerank_results.id, "left").drop("id")
        df_year = df_year.withColumnRenamed("pagerank", "origin_pagerank")

        df_year = df_year.join(pagerank_results, df_year.DEST == pagerank_results.id, "left").drop("id")
        df_year = df_year.withColumnRenamed("pagerank", "dest_pagerank")

        # Ensure consistent schema by aligning columns
        if final_df is None:
            final_df = df_year  # Initialize final_df with the first year's data
        else:
            # Add missing columns in df_year or final_df to match their schemas
            for col_name in set(final_df.columns) - set(df_year.columns):
                df_year = df_year.withColumn(col_name, F.lit(None))
            for col_name in set(df_year.columns) - set(final_df.columns):
                final_df = final_df.withColumn(col_name, F.lit(None))
            
            # Append the result to the final DataFrame
            final_df = final_df.unionByName(df_year)

    return final_df


# COMMAND ----------

# MAGIC %md
# MAGIC #### Normalization of Data

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

def normalize_numerical_features(df):
    """
    Normalize numerical features in a DataFrame.
    """
    # Assemble the numerical columns into a single vector column
    numerical_cols = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight']
    numerical_cols.extend(['PrevFlightDelay', 'origin_pagerank', 'dest_pagerank'])
    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
    df = assembler.transform(df)
    
    # Normalize the numerical features
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    return df

# COMMAND ----------

def preprocess_and_engineer_features(df):
    df = preprocess_and_engineer_features_stateless(df)
    df = preprocess_and_engineer_features_stateful(df)
    return df

def preprocess_and_engineer_features_stateless(df):
    """
    row by row transformation - no leakage risk
    """
    df_clean = filter_relevant_flights(df)
    df_clean = drop_duplicates_records(df)
    df_clean = select_columns(df_clean)
    df_clean = create_vert_obscuration_flag(df_clean)
    df_clean = convert_string_to_time(df_clean)
    df_clean = create_departure_bins(df_clean)
    df_clean = generate_num_of_hours_since_last_flight(df_clean)
    df_clean = generate_previous_flight_delay_minutes(df_clean)
    df_clean = limit_delay_tail(df_clean)
    df_clean = generate_weather_boolean(df_clean)
    df_clean = add_holiday_flag(df_clean, "FL_DATE")
    # df_clean = add_pagerank_features_blocks(df_clean)
    dropped_count, df_clean = count_and_drop_rows_with_na(df_clean)
    df_clean = one_hot_encode(df_clean) # move
    return df_clean


def preprocess_and_engineer_features_stateful(df):
    """
    transformations that depends on previous data (statful)
    """
    df_clean = add_pagerank_features_blocks(df)
    # df_clean = one_hot_encode(df_clean)
    df_normalized = normalize_numerical_features(df_clean)
    return df_normalized

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving Changes to Storage

# COMMAND ----------

def write_to_blob(df, filename):
    """  Write a Spark DataFrame to a blob storage container """
    df.write.parquet(f"{team_blob_url}/Phase_3/{filename}", mode="overwrite")

# COMMAND ----------

# write normalized 3 months of data to blob storage
df_3m_normalized = preprocess_and_engineer_features(df_flights_weather_3m)
display(df_3m_normalized)
write_to_blob(df_3m_normalized, "Cleaned_Data_3m")

# COMMAND ----------

# write normalized one year of data to blob storage
df_1y_normalized = preprocess_and_engineer_features(df_flights_weather_1y)
write_to_blob(df_1y_normalized, "Cleaned_Data_1y")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split dataset for training and testing
# MAGIC to prevent leakage
# MAGIC
# MAGIC 5y dataset split:
# MAGIC - training and validation dataset (including cross validation)=2015, 2016, 2017 , 2018 (4y)
# MAGIC - data as a blind test set = 2019 (test)
# MAGIC

# COMMAND ----------

# apply stateless feature engineering to all 5 years of data
df_flights_weather_5y = preprocess_and_engineer_features_stateless(df_flights_weather_5y).cache()

# COMMAND ----------

# Filter the training dataset (2015, 2016, 2017, 2018)
df_train_4y = df_flights_weather_5y.filter(df_flights_weather_5y.YEAR != 2019).cache()

# apply stateless feature engineering
df_train_4y = preprocess_and_engineer_features_stateful(df_train_4y)

# write 4y training dataset to blob storage
write_to_blob(df_train_4y, "Cleaned_Data_5y/train_4y")

# COMMAND ----------

# Filter the training dataset (2015, 2016, 2017, 2018)
df_test_1y = df_flights_weather_5y.filter(df_flights_weather_5y.YEAR == 2019).cache()

# apply stateless feature engineering
df_test_1y = preprocess_and_engineer_features_stateful(df_test_1y)

# use pagerank from years 2015-2018 as the initial weights for the 2019 test dataset

# Group by airport ID and calculate the average PageRank score over 2015-2018
pagerank_aggregated = df_train_4y.groupBy("id").agg(F.mean("pagerank").alias("aggregated_pagerank"))
# Join the aggregated PageRank scores with the 2019 data for the origin airport
df_test_1y = df_test_1y.join(pagerank_aggregated, df_test_1y.ORIGIN == pagerank_aggregated.id, "left").drop("id")
df_test_1y = df_test_1y.withColumnRenamed("aggregated_pagerank", "origin_pagerank")

# Join the aggregated PageRank scores with the 2019 data for the destination airport
df_test_1y = df_test_1y.join(pagerank_aggregated, df_test_1y.DEST == pagerank_aggregated.id, "left").drop("id")
df_test_1y = df_test_1y.withColumnRenamed("aggregated_pagerank", "dest_pagerank")

# write 1y test dataset to blob storage
write_to_blob(df_test_1y, "Cleaned_Data_5y/test_1y")

# COMMAND ----------

# list output folders in blob storage  
# display(dbutils.fs.ls(f"{team_blob_url}/Phase_3"))
display(dbutils.fs.ls(f"{team_blob_url}/Phase_3/Cleaned_Data_5y"))

# COMMAND ----------

# test read data from blob storage
display(spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y").limit(100))

# COMMAND ----------

# test read data from blob storage
display(spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/test_1y").limit(100))