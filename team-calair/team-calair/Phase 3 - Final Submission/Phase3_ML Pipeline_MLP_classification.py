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

from pyspark.sql.functions import when, col

# Load dataset
#parquet_file_path = f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y"
#clean_data = spark.read.parquet(parquet_file_path)

df_train = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y")
df_test = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/test_1y")

# Create the binary output column 'delay_or_not'
df_train = df_train.withColumn('delay_or_not', when(col('dep_capped_delay') > 15, 1).otherwise(0)).cache()
df_test = df_test.withColumn('delay_or_not', when(col('dep_capped_delay') > 15, 1).otherwise(0)).cache()


# COMMAND ----------

display(df_train.limit(10))

# COMMAND ----------

selected_columns_df_train = df_train.select(*X)
display(selected_columns_df_train.limit(10))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

#Define X set
categoricals = ['DAY_OF_WEEK_onehot','Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerics = ['scaled_features']
booleans = ['is_extreme_weather', 'is_holiday', 'is_vert_obscured']

#Make a list of column names to tell us what is in numerics
numerical_cols = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay','origin_pagerank', 'dest_pagerank']

X = categoricals + numerics + booleans
Y = 'delay_or_not'

# Assemble the feature vector
assembler = VectorAssembler(inputCols=X, outputCol='features')
assembled_df_train = assembler.transform(df_train).select('features', col(Y).alias('label'), 'FL_DATE')
assembled_df_test = assembler.transform(df_test).select('features', col(Y).alias('label'), 'FL_DATE')

# COMMAND ----------

display(assembled_df_train)

# COMMAND ----------

from pyspark.sql.functions import when, col
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import matplotlib.pyplot as plt


# COMMAND ----------

def evaluate_f3_score(TP, TN, FP, FN):
    """
    Calculates F3 Score
    """
    return TP / (TP + 0.1 * FP + 0.9 * FN)

# COMMAND ----------

# Define a function for time-series cross-validation on a ROLLING BASIS
def time_series_cv(data, date_col, model, metric, folds=5):
    data = data.orderBy(date_col)
    train_ratio = 1/(folds+1)
    total_rows = data.count()
    
    print('adding row number column')
    # Add a row number column
    window_spec = Window.orderBy(date_col)
    data_with_row_number = data.withColumn("row_num", row_number().over(window_spec))

    metrics = []
    coefficients = []

    for fold in range(1,folds+1):

        print(f'Starting CV round {fold}.')

        # Define the training and validation split
        train_end = int(fold * total_rows * train_ratio)
        validation_end = train_end + int(total_rows * train_ratio)

        # Split the data into training and validation sets
        train_data = data_with_row_number.filter(col("row_num") <= train_end)
        validation_data = data_with_row_number.filter((col("row_num") > train_end) & (col("row_num") <= validation_end))
        
        # Train the model
        temp_model = model.fit(train_data)
        
        # Make predictions
        predictions = temp_model.transform(validation_data)
        """
        # Extract True Labels and predictions
        preds_and_labels = predictions.select("prediction", "label").rdd

        # Compute confusion matrix components
        matrix = preds_and_labels.map(lambda pl: ((pl[0], pl[1]), 1)).reduceByKey(lambda a, b: a + b).collectAsMap()

        # Extract the components
        TP = matrix.get((1.0, 1.0), 0)
        TN = matrix.get((0.0, 0.0), 0)
        FP = matrix.get((1.0, 0.0), 0)
        FN = matrix.get((0.0, 1.0), 0)

        # Calculate F3 Score
        f3_score = evaluate_f3_score(TP, TN, FP, FN)
        
        # Evaluate the model
        #metric_value = evaluator.evaluate(predictions)
        metrics.append(f3_score)
        print(f"Fold {fold}: {metric} = {f3_score}")
        """

        precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
        recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


        # Calculate precision and recall
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)

        # Calculate F3 score
        beta = 3
        f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        print(f"Fold {fold}: {metric} = {f3_score}")

        metrics.append(f3_score)
        # Store Coefficients for Optional Analysis
        #temp_coefficients = temp_model.coefficients.toArray()
        #coefficients.append(temp_coefficients)

        # Store the best model
        if f3_score >= max(metrics):
            best_cvModel = temp_model

    average_f3 = sum(metrics) / len(metrics)
    print(f"Average F3: {average_f3}")

    #get the best model
    best_result = max(metrics)
    print(f'Best Model is from fold {metrics.index(best_result)+1} with F3 {best_result}.')

    #return metrics, coefficients, best_cvModel
    return metrics, best_cvModel


# COMMAND ----------

# Determine the input size (number of features)
input_size = len(assembled_df_train.select("features").first()[0])

# Define the layers for the neural network
layers = [input_size, 2, 2]

# Initialize the MultilayerPerceptronClassifier
mlp_model = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol='features', labelCol='label')

#cv_coefficients
cv_metrics, best_cvModel = time_series_cv(assembled_df_train, 'FL_DATE', mlp_model, metric = "F3")

# COMMAND ----------

predictions = best_cvModel.transform(assembled_df_test)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# Calculate precision and recall
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Calculate F3 score
beta = 3
f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print(f"Model 1 Final Score = {f3_score}")

# COMMAND ----------

# Determine the input size (number of features)
input_size = len(assembled_df_train.select("features").first()[0])

# Define the layers for the neural network
layers = [input_size, 4, 2]

# Initialize the MultilayerPerceptronClassifier
mlp_model = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol='features', labelCol='label')

#cv_coefficients
cv_metrics_2, best_cvModel_2 = time_series_cv(assembled_df_train, 'FL_DATE', mlp_model, metric = "F3")

# COMMAND ----------

predictions = best_cvModel_2.transform(assembled_df_test)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# Calculate precision and recall
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Calculate F3 score
beta = 3
f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print(f"Model 2 Final Score = {f3_score}")

# COMMAND ----------

# Determine the input size (number of features)
input_size = len(assembled_df_train.select("features").first()[0])

# Define the layers for the neural network
layers = [input_size, 2, 2, 2]

# Initialize the MultilayerPerceptronClassifier
mlp_model = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol='features', labelCol='label')

#cv_coefficients
cv_metrics_3, best_cvModel_3 = time_series_cv(assembled_df_train, 'FL_DATE', mlp_model, metric = "F3")

# COMMAND ----------

predictions = best_cvModel_3.transform(assembled_df_test)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# Calculate precision and recall
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Calculate F3 score
beta = 3
f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print(f"Model 3 Final Score = {f3_score}")

# COMMAND ----------

# Determine the input size (number of features)
input_size = len(assembled_df_train.select("features").first()[0])

# Define the layers for the neural network
layers = [input_size, 4, 2, 2]

# Initialize the MultilayerPerceptronClassifier
mlp_model = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol='features', labelCol='label')

#cv_coefficients
cv_metrics_4, best_cvModel_4 = time_series_cv(assembled_df_train, 'FL_DATE', mlp_model, metric = "F3")

# COMMAND ----------

predictions = best_cvModel_4.transform(assembled_df_test)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# Calculate precision and recall
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Calculate F3 score
beta = 3
f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print(f"Model 4 Final Score = {f3_score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coeffcients (Ignore its useless)

# COMMAND ----------

import pandas as pd

weight = best_cvModel_2.weights.toArray()
columns = assembled_df_test.schema["features"].metadata["ml_attr"]["attrs"]["numeric"] + assembled_df_test.schema["features"].metadata["ml_attr"]["attrs"]["binary"]
feature_names = [col["name"] for col in columns]
weights_with_columns = dict(zip(feature_names, weight))

# Convert to pandas DataFrame
df = pd.DataFrame(list(weights_with_columns.items()), columns=['Feature', 'Weight'])

# Calculate absolute weights and sort
df['AbsoluteWeight'] = df['Weight'].abs()
df_sorted = df.sort_values(by='AbsoluteWeight', ascending=False)

# Get top 10 and bottom 10 attributes by absolute weight
top_10_df = df_sorted.head(10)
bottom_10_df = df_sorted.tail(10)

display(top_10_df)
display(bottom_10_df)

# COMMAND ----------

#best_cvModel_2.coefficients.toArray()

weight = best_cvModel_2.weights.toArray()
columns = assembled_df_test.schema["features"].metadata["ml_attr"]["attrs"]["numeric"] + assembled_df_test.schema["features"].metadata["ml_attr"]["attrs"]["binary"]
feature_names = [col["name"] for col in columns]
weights_with_columns = dict(zip(feature_names, weight))
display(weights_with_columns)