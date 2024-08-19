# Databricks notebook source
# MAGIC %md
# MAGIC ### Linear Regression with Time-Series Rolling Cross Validation on 3-mo Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ML Pipeline inspired by and some code sourced from: https://pages.databricks.com/rs/094-YMS-629/images/02-Delta%20Lake%20Workshop%20-%20Including%20ML.html

# COMMAND ----------

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

#Get Feature Engineered dataset from blob storage

# Define the path to the Parquet files
parquet_file_path = f"{team_blob_url}/Phase_2/Cleaned_Data_1y"

# Read the Parquet files into a Spark DataFrame
clean_data = spark.read.parquet(parquet_file_path)

# Display the DataFrame
display(clean_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Out Data

# COMMAND ----------

clean_data.count()

# COMMAND ----------

import matplotlib.pyplot as plt

departure_delay_pd = clean_data.select('DEP_DELAY_NEW').toPandas()

plt.hist(departure_delay_pd, bins=50)

# COMMAND ----------

#engineered capped delay (to eliminate super long tail)

departure_capped_delay_pd = clean_data.select('dep_capped_delay').toPandas()

plt.hist(departure_capped_delay_pd, bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Response and Predictor Variables

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict delays")

#Define Y
Y = 'dep_capped_delay'

#Define X set
categoricals = ['DAY_OF_WEEK_onehot','Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerics = ['scaled_features']
booleans = ['is_extreme_weather', 'is_holiday', 'is_vert_obscured']

#Make a list of column names to tell us what is in numerics
num_col = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay']

#Compile X set
X = categoricals + numerics + booleans

flight_data = clean_data.select(X + [Y] + ['FL_DATE'] + ['DEP_DELAY_NEW'])

# COMMAND ----------

#back out the column names for one-hot since we can't retrieve them from feature engineered dataset

feature_list = []

for column in categoricals:
    metadata = clean_data.schema[column].metadata
    metadata = metadata['ml_attr']['attrs']['binary']
    # print(f"Metadata for {column}: {metadata}")
    idx_name = [column + '_' + x['name'] for x in metadata]
    feature_list.extend(idx_name)

feature_list.extend(num_col)
feature_list.extend(booleans)

# COMMAND ----------

len(feature_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Simple Linear Models
# MAGIC
# MAGIC Because of the distrubion of the data, it makes sense to use GLM instead of linear regression framework in Apache Spark. [More](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#linear-regression).

# COMMAND ----------

#import packages
# from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

#Use VectorAssembler to compile all data into a single vector column
va = VectorAssembler(inputCols = X, outputCol = 'features')
va_df = va.transform(flight_data)
va_df = va_df.select(['features', 'dep_capped_delay', 'FL_DATE', 'DEP_DELAY_NEW'])

# Choose a cutoff date for splitting the data
cutoff_date = "2015-10-31"

#Split by Month for now.. need to figure out how to split by date.
train = va_df.filter(va_df.FL_DATE <= cutoff_date).cache()
valid = va_df.filter(va_df.FL_DATE > cutoff_date).cache()

print(train.count())
print(valid.count())

# COMMAND ----------

# Using linear regression 
lr_model = LinearRegression(featuresCol = "features", \
                        labelCol = 'dep_capped_delay')

# COMMAND ----------

lr_model = lr_model.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation Linear Model

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

# COMMAND ----------

# Initialize new linear model for CV
cv_lr_model = LinearRegression(featuresCol = "features", \
                        labelCol = 'dep_capped_delay')

#Initialize Evaluator for CV
cv_evaluator = RegressionEvaluator(
    labelCol="dep_capped_delay", predictionCol="prediction", metricName="rmse")

# COMMAND ----------

# Define a function for time-series cross-validation on a ROLLING BASIS
def time_series_cv(data, date_col, model, evaluator, metric, folds=5):
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
        
        # Evaluate the model
        metric_value = evaluator.evaluate(predictions)
        metrics.append(metric_value)
        print(f"Fold {fold}: {metric} = {metric_value}")
        
        # Store Coefficients for Optional Analysis
        temp_coefficients = temp_model.coefficients.toArray()
        coefficients.append(temp_coefficients)

        # Store the best model
        if metric_value >= max(metrics):
            best_cvModel = temp_model

    average_rmse = sum(metrics) / len(metrics)
    print(f"Average RMSE: {average_rmse}")

    #get the best model
    best_result = min(metrics)
    print(f'Best Model is from fold {metrics.index(best_result)+1} with RMSE {best_result}.')

    return metrics, coefficients, best_cvModel


# COMMAND ----------

# Perform time-series cross-validation
cv_metrics, cv_coefficients, best_cvModel = time_series_cv(train, 'FL_DATE', cv_lr_model, cv_evaluator, 'rmse')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summarize Models

# COMMAND ----------

print(f'LR MAE: {lr_model.summary.meanAbsoluteError}')
print(f'LR RMSE: {lr_model.summary.rootMeanSquaredError}')
print(f'CV MAE: {best_cvModel.summary.meanAbsoluteError}')
print(f'CV RMSE: {best_cvModel.summary.rootMeanSquaredError}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Baseline

# COMMAND ----------

from pyspark.sql.functions import lit, mean, broadcast

# Calculate the average delay
avg_delay_df = valid.select(mean('DEP_DELAY_NEW').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df = broadcast(avg_delay_df)

# Join the average delay with the valid DataFrame
valid_with_avg = valid.crossJoin(avg_delay_df)

# Calculate the average delay
avg_delay_df_train = train.select(mean('DEP_DELAY_NEW').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df_train = broadcast(avg_delay_df_train)

# Join the average delay with the valid DataFrame
train_with_avg = valid.crossJoin(avg_delay_df_train)

# COMMAND ----------

evaluator = RegressionEvaluator(
    labelCol="DEP_DELAY_NEW", predictionCol="avg_delay", metricName="rmse")
baseline_rmse_valid = evaluator.evaluate(valid_with_avg)
print("Root Mean Squared Error (RMSE) on data: /n")
print(f'Baseline valid: {baseline_rmse_valid}')

baseline_rmse_train = evaluator.evaluate(train_with_avg)
print(f'Baseline train: {baseline_rmse_train}')

# COMMAND ----------

evaluator = RegressionEvaluator(
    labelCol="DEP_DELAY_NEW", predictionCol="avg_delay", metricName="mae")
baseline_mae_valid = evaluator.evaluate(valid_with_avg)
print("MAE on  data: /n")
print(f'Baseline valid: {baseline_mae_valid}')

baseline_mae_train = evaluator.evaluate(train_with_avg)
print(f'Baseline train: {baseline_mae_train}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Model Predictions

# COMMAND ----------

lr_pred = lr_model.transform(valid)
cv_pred = best_cvModel.transform(valid)

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="dep_capped_delay", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lr_pred)
cv_rmse = evaluator.evaluate(cv_pred)
print("Root Mean Squared Error (RMSE) on test data: /n")
print(f'Linear Regression: {lr_rmse}')
print(f'Cross Validation: {cv_rmse}')

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="dep_capped_delay", predictionCol="prediction", metricName="mae")
lr_mae = evaluator.evaluate(lr_pred)
cv_mae = evaluator.evaluate(cv_pred)
print("Mean Average Error (MAE) on test data: /n")
print(f'Linear Regression: {lr_mae}')
print(f'Cross Validation: {cv_mae}')

# COMMAND ----------

### Feature Importance

# Extract feature importances (coefficients)
lr_coefficients = lr_model.coefficients.toArray()
cv_coefficients = best_cvModel.coefficients.toArray()
feature_importances = list(zip(feature_list, abs(lr_coefficients), abs(cv_coefficients)))


# Sort the list by importance in descending order
lr_sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
cv_sorted_feature_importances = sorted(feature_importances, key=lambda x: x[2], reverse=True)

# Select the top 10 features
lr_top_10_features = lr_sorted_feature_importances[:10]
cv_top_10_features = cv_sorted_feature_importances[:10]

# Select the top 10 features
lr_bottom_10_features = lr_sorted_feature_importances[-10:]
cv_bottom_10_features = cv_sorted_feature_importances[-10:]

# COMMAND ----------

lr_top_10_features

# COMMAND ----------

lr_bottom_10_features

# COMMAND ----------

cv_top_10_features

# COMMAND ----------

cv_bottom_10_features

# COMMAND ----------

import matplotlib.pyplot as plt

# Function to plot top 10 features
def plot_top_features(top_features, index, title):
    features, importances = zip(*[(top_feature[0], top_feature[index]) for top_feature in top_features])
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances, color='navy')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    display(plt.show())

# Plotting top 10 features for each model
plot_top_features(lr_top_10_features, 1,'Top 10 Features for Linear Regression Model')
plot_top_features(cv_top_10_features, 2, 'Top 10 Features for Cross-Validated Model')

# COMMAND ----------

# Plotting top 10 features for each model
plot_top_features(lr_bottom_10_features, 1,'Bottom 10 Features for Linear Regression Model')
plot_top_features(cv_bottom_10_features, 2, 'Bottom 10 Features for Cross-Validated Model')

# COMMAND ----------

# MAGIC %md
# MAGIC F3 Evaluation

# COMMAND ----------

# Add delay_no_delay column based on the prediction value (0 for delay, 1 for no delay) for classification and f3 scope calculation  

from pyspark.sql.functions import when
from pyspark.sql.types import DoubleType

#baseline augment data for classification
valid_with_avg = valid_with_avg.withColumn("delay_binary_pred", when(valid_with_avg["avg_delay"] > 15, 1).otherwise(0).cast(DoubleType()))
valid_with_avg = valid_with_avg.withColumn("delay_binary_label", when(valid_with_avg["DEP_DELAY_NEW"] > 15, 1).otherwise(0).cast(DoubleType()))

train_with_avg = train_with_avg.withColumn("delay_binary_pred", when(train_with_avg["avg_delay"] > 15, 1).otherwise(0).cast(DoubleType()))
train_with_avg = train_with_avg.withColumn("delay_binary_label", when(train_with_avg["DEP_DELAY_NEW"] > 15, 1).otherwise(0).cast(DoubleType()))
# #linear model augment data for classification
# lr_pred = lr_pred.withColumn("delay_binary_pred", when(lr_pred["prediction"] > 15, 1).otherwise(0).cast(DoubleType()))
# lr_pred = lr_pred.withColumn("delay_binary_label", when(lr_pred["dep_capped_delay"] > 15, 1).otherwise(0).cast(DoubleType()))

# #CV augment data for classification
# cv_pred = cv_pred.withColumn("delay_binary_pred", when(cv_pred["prediction"] > 15, 1).otherwise(0).cast(DoubleType()))
# cv_pred = cv_pred.withColumn("delay_binary_label", when(cv_pred["dep_capped_delay"] > 15, 1).otherwise(0).cast(DoubleType()))

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluators for precision and recall
precision_evaluator = MulticlassClassificationEvaluator(labelCol="delay_binary_label", predictionCol="delay_binary_pred", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="delay_binary_label", predictionCol="delay_binary_pred", metricName="weightedRecall")

# Calculate baseline precision and recall
valid_baseline_precision = precision_evaluator.evaluate(valid_with_avg)
valid_baseline_recall = recall_evaluator.evaluate(valid_with_avg)
train_baseline_precision = precision_evaluator.evaluate(train_with_avg)
train_baseline_recall = recall_evaluator.evaluate(train_with_avg)

# Calculate linear regression precision and recall
# lr_precision = precision_evaluator.evaluate(lr_pred)
# lr_recall = recall_evaluator.evaluate(lr_pred)

# Calculate Baseline F3 score
beta = 3
valid_f3_score = (1 + beta**2) * (valid_baseline_precision * valid_baseline_recall) / ((beta**2 * valid_baseline_precision) + valid_baseline_recall)
print(f"Baseline F3 Score valid: {valid_f3_score}")

train_f3_score = (1 + beta**2) * (train_baseline_precision * train_baseline_recall) / ((beta**2 * train_baseline_precision) + train_baseline_recall)
print(f"Baseline F3 Score Train: {train_f3_score}")

# # Calculate Linear Regression F3 score
# beta = 3
# f3_score = (1 + beta**2) * (lr_precision * lr_recall) / ((beta**2 * lr_precision) + lr_recall)
# print(f"Linear Model F3 Score: {f3_score}")

# COMMAND ----------

