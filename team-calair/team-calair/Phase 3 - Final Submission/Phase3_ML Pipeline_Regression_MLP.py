# Databricks notebook source
#!pip install elephas
#%restart_python


# COMMAND ----------

#Connect to Cloud Database

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

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col

# COMMAND ----------


# Load dataset

df_train = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y")
df_test = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/test_1y")

# COMMAND ----------

#Define X set
categoricals = ['DAY_OF_WEEK_onehot','Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerics = ['scaled_features']
booleans = ['is_extreme_weather', 'is_holiday', 'is_vert_obscured']

#Make a list of column names to tell us what is in numerics
numerical_cols = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay','origin_pagerank', 'dest_pagerank']

X = categoricals + numerics + booleans
Y = 'dep_capped_delay'

assembler = VectorAssembler(inputCols=X, outputCol='features')
assembled_df_train = assembler.transform(df_train).select('features', col(Y).alias('label'), 'FL_DATE')
assembled_df_test = assembler.transform(df_test).select('features', col(Y).alias('label'), 'FL_DATE')

# COMMAND ----------

print(assembled_df_test.select("features").first()[0])

# COMMAND ----------

len(assembled_df_test.select("features").first()[0])

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

input_size = 758 #len(assembled_df_test.select("features").first()[0])

# Model
model = Sequential()
model.add(Dense(2, input_shape=(input_size,)))
model.add(Activation('relu'))
model.add(Dense(1))


# COMMAND ----------

from elephas.ml_model import ElephasEstimator
from elephas.ml.adapter import to_data_frame
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf

adam = Adam(learning_rate=0.001)  # Default is 0.001, change as needed
serialized_adam = tf.keras.optimizers.serialize(adam)


# Convert Keras model to Elephas estimator
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_json())
estimator.set_optimizer_config(serialized_adam)
estimator.set_mode("synchronous")
estimator.set_loss("root_mean_squared_error")
estimator.set_metrics(['rmse'])
estimator.set_epochs(10)
estimator.set_batch_size(5)
estimator.set_verbosity(1)
estimator.set_num_workers(4)  # Adjust based on your cluster setup


# COMMAND ----------

#Trying to fit on the test set because its smaller
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[estimator])
fitted_pipeline = pipeline.fit(assembled_df_test)

# COMMAND ----------

