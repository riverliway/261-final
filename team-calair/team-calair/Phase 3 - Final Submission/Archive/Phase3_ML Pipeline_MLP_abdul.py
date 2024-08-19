# Databricks notebook source
# MAGIC %md
# MAGIC ## Access Dataset

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

# MAGIC %md
# MAGIC ### MultilayerPerceptronClassifier Prototype

# COMMAND ----------

# Load dataset
parquet_file_path = f"{team_blob_url}/Phase_3/Cleaned_Data_1y"
clean_data = spark.read.parquet(parquet_file_path)

# Create the binary output column 'delay_or_not'
clean_data = clean_data.withColumn('delay_or_not', when(col('dep_capped_delay') > 15, 1).otherwise(0))
clean_data.cache()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Define the feature columns
categoricals = ['DAY_OF_WEEK_onehot', 'Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerical_cols = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay', 'origin_pagerank', 'dest_pagerank']

booleans = ['is_extreme_weather', 'is_holiday', 'is_vert_obscured']
X = categoricals + numerical_cols + booleans
Y = 'delay_or_not'

# Assemble the feature vector
assembler = VectorAssembler(inputCols=X, outputCol='features')
assembled_data = assembler.transform(clean_data).select('features', col(Y).alias('label'))

# COMMAND ----------

from pyspark.sql.functions import when, col
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Perform a random split
train, valid = assembled_data.randomSplit([0.8, 0.2], seed=42)

# Determine the input size (number of features)
input_size = len(assembled_data.select("features").first()[0])

# Define the layers for the neural network
layers = [input_size, 5, 4, 2]

# Initialize the MultilayerPerceptronClassifier
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol='features', labelCol='label')

# Train the model
model = trainer.fit(train)

# Make predictions
predictions = model.transform(valid)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='probability', metricName='areaUnderROC')
roc_auc = evaluator.evaluate(predictions)
print(f"Area Under ROC on validation data: {roc_auc}")

evaluator_pr = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='probability', metricName='areaUnderPR')
pr_auc = evaluator_pr.evaluate(predictions)
print(f"Area Under PR on validation data: {pr_auc}")