# Databricks notebook source
# MAGIC %md
# MAGIC # 3 Month

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

def get_data_from_cloud (dataset: Literal['OTPW_3M_2015.csv', 'OTPW_1Y_2015.csv']) -> DataFrame:
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

show_metadata(df_flights_weather_3m)


# COMMAND ----------

def find_empty_values_per_column (df: DataFrame) -> None:
    """
    Finds the number of empty values in every column
    WARNING: Takes a long time to complete ~20mins
    """
    total = df.count()

    for column in df.columns:
        num_missing = df.filter(f'{column} IS NULL').count()
        percent = f"({(num_missing / total) * 100:.2f}%)" if num_missing > 0 else ""
        print(f"{column}: {num_missing} {percent}")

find_empty_values_per_column(df_flights_weather_3m)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, DoubleType, DateType, FloatType, DoubleType
from pyspark.sql.functions import to_timestamp, lpad, concat_ws, when, col, date_add, hour, lag, unix_timestamp
from pyspark.sql import Window


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


df_flights_weather_3m_clean = filter_relevant_flights(df_flights_weather_3m)

# COMMAND ----------

columns = [
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
    "HourlySkyConditions"
]

df_flights_weather_3m_clean = df_flights_weather_3m.select(columns)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Assigning Data Types

# COMMAND ----------

# Ensure the desired data types for each column
data_types = {
    "DAY_OF_MONTH": IntegerType(),
    "DAY_OF_WEEK": IntegerType(),
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
    "HourlyWindSpeed": DoubleType()
}

# Apply the data types to each column
for column, dtype in data_types.items():
    df_flights_weather_3m_clean = df_flights_weather_3m_clean.withColumn(column, col(column).cast(dtype))

# Show the schema to verify the changes
df_flights_weather_3m_clean.printSchema()

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

df_flights_weather_3m_clean = create_vert_obscuration_flag(df_flights_weather_3m_clean)



# COMMAND ----------

def convert_string_to_time(df):
    """
    Convert the CRS_DEP_TIME and CRS_ARR_TIME columns to a timestamp format.
    """
    df = df.withColumn("CRS_DEP_TIME", to_timestamp(concat_ws(' ', col("FL_DATE"), lpad(col("CRS_DEP_TIME"), 4, '0')), 'yyyy-MM-dd HHmm'))
    df = df.withColumn("CRS_ARR_TIME", to_timestamp(concat_ws(' ', col("FL_DATE"), lpad(col("CRS_ARR_TIME"), 4, '0')), 'yyyy-MM-dd HHmm'))
    df = df.withColumn("CRS_ARR_TIME", when(col("CRS_ARR_TIME") < col("CRS_DEP_TIME"), date_add(col("CRS_ARR_TIME"), 1)).otherwise(col("CRS_ARR_TIME")))
    return df

df_flights_weather_3m_clean = convert_string_to_time(df_flights_weather_3m_clean)

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

df_flights_weather_3m_clean = create_departure_bins(df_flights_weather_3m_clean)

# COMMAND ----------

def generate_num_of_hours_since_last_flight(df):
    """
    Generate a new column that will count the number of hours since the last flight for a particular airline/airport
    """
    
    window_spec = Window.partitionBy("TAIL_NUM").orderBy("CRS_ARR_TIME")
    df = df.withColumn("PrevFlightArrive", lag(col("CRS_ARR_TIME")).over(window_spec))
    df = df.withColumn("HoursSinceLastFlight", (unix_timestamp(col("CRS_DEP_TIME")) - unix_timestamp(col("PrevFlightArrive"))) / 3600)
    df = df.fillna({"HoursSinceLastFlight": 0})
    return df

df_flights_weather_3m_clean = generate_num_of_hours_since_last_flight(df_flights_weather_3m_clean)

# COMMAND ----------

def generate_previous_flight_delay_minutes(df):
    """
    Generate a new column that will count the number of hours since the last flight for a particular airline/airport
    """
    
    window_spec = Window.partitionBy("TAIL_NUM").orderBy("CRS_ARR_TIME")
    df = df.withColumn("PrevFlightDelay", lag(col("DEP_DELAY_NEW")).over(window_spec))
    df = df.fillna({"PrevFlightDelay": 0})
    return df

df_flights_weather_3m_clean = generate_previous_flight_delay_minutes(df_flights_weather_3m_clean)

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

dropped_count, df_flights_weather_3m_clean = count_and_drop_rows_with_na(df_flights_weather_3m_clean)
print(dropped_count)

# COMMAND ----------

categorical_columns = [field.name for field in df_flights_weather_3m_clean.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df_flights_weather_3m_clean.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))]

print(categorical_columns)
print(numerical_columns)

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

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def one_hot_encode(df, categorical_columns):
    """
    Convert categorical columns to one-hot encoded columns.
    """
    stages = []
    for col in categorical_columns:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_onehot")
        stages += [indexer, encoder]
    
    pipeline = Pipeline(stages=stages)
    df = pipeline.fit(df).transform(df)
    
    return df
categorical_columns = ['ORIGIN', 'DEST', 'OP_CARRIER', 'Departure_Time_Buckets']
df_flights_weather_3m_clean = one_hot_encode(df_flights_weather_3m_clean, categorical_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Normalization of Data

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

def normalize_numerical_features(df, numerical_cols):
    """
    Normalize numerical features in a DataFrame.
    """
    # Assemble the numerical columns into a single vector column
    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
    df = assembler.transform(df)
    
    # Normalize the numerical features
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    return df

num_col = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay']
df_normalized = normalize_numerical_features(df_flights_weather_3m_clean, num_col)
display(df_normalized)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving Changes to Storage

# COMMAND ----------


def write_to_blob(df):
    """  Write a Spark DataFrame to a blob storage container """
    blob_container  = "ds261-final-project-team-2-2"       # The name of your container created in https://portal.azure.com
    storage_account = "ds261team"  # The name of your Storage account created in https://portal.azure.com
    secret_scope    = "261_team_2_2_scope"           # The name of the scope created in your local computer using the Databricks CLI
    secret_key      = "261_team_2_2_key"             # The name of the secret key created in your local computer using the Databricks CLI
    team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket
    df.write.parquet(f"{team_blob_url}/Phase_2/Cleaned_Data", mode="overwrite")
write_to_blob(df_normalized)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Implementing Lasso

# COMMAND ----------

"""
from pyspark.ml.regression import LinearRegression

# Use Lasso with elasticNetParam = 1.0 for LASSO regularization
lasso = LinearRegression(featuresCol='features', labelCol='DEP_DELAY_NEW', elasticNetParam=1.0)
lasso_model = lasso.fit(data)
predictions_lasso = lasso_model.transform(data)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # 23 Month
# MAGIC

# COMMAND ----------

df_flights_weather_1y = get_data_from_cloud('OTPW_1Y_2015.csv')

# COMMAND ----------

