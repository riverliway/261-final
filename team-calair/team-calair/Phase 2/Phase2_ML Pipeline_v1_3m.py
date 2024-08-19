# Databricks notebook source
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
parquet_file_path = f"{team_blob_url}/Phase_2/Cleaned_Data"

# Read the Parquet files into a Spark DataFrame
clean_data = spark.read.parquet(parquet_file_path)

# Display the DataFrame
display(clean_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Out Data

# COMMAND ----------

import matplotlib.pyplot as plt

departure_delay_pd = clean_data.select('DEP_DELAY_NEW').toPandas()

plt.hist(departure_delay_pd, bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Response and Predictor Variables

# COMMAND ----------

clean_data.count()

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict delays")

#Define Y
Y = "DEP_DELAY_NEW"

#Define X set
categoricals = ['Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerics = ['scaled_features']

#TAKING OUT THE LAST FLIGHT DELAY
X = categoricals + numerics[:-1]

flight_data = clean_data.select(X + [Y] + ['MONTH'])

# COMMAND ----------

#back out the column names for one-hot since we can't retrieve them from feature engineered dataset

feature_list = []

for column in categoricals:
    metadata = clean_data.schema[column].metadata
    metadata = metadata['ml_attr']['attrs']['binary']
    # print(f"Metadata for {column}: {metadata}")
    idx_name = [column + '_' + x['name'] for x in metadata]
    feature_list.extend(idx_name)

num_col = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight'] #, 'PrevFlightDelay']
#weather data not particularly helpful in the prediction >> sparse features on *extreme weather events* may be more useful
#include prior flight delay
#include holidays / seasonality

feature_list.extend(num_col)

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
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

va = VectorAssembler(inputCols = X, outputCol = 'features')
va_df = va.transform(flight_data)
va_df = va_df.select(['features', 'DEP_DELAY_NEW', 'MONTH'])

#Split by Month for now.. need to figure out how to split by date.
train = va_df.filter(va_df.MONTH <= 2).cache()
valid = va_df.filter(va_df.MONTH > 2).cache()

# train.count()
# valid.count()

# COMMAND ----------

# Using linear regression 
lr_model = LinearRegression(featuresCol = "features", \
                        labelCol = 'DEP_DELAY_NEW')

# Using Generalized linear regression w/ tweedie based on data skew
glr_model = GeneralizedLinearRegression(family = 'tweedie', \
                                    featuresCol = "features", \
                                    labelCol = 'DEP_DELAY_NEW')

# COMMAND ----------

lr_model = lr_model.fit(train)
# glr_model = glr_model.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation Linear Model

# COMMAND ----------

# Using linear regression 
cv_lr_model = LinearRegression(featuresCol = "features", \
                        labelCol = 'DEP_DELAY_NEW')

# Create a parameter grid
paramGrid = ParamGridBuilder()\
    .addGrid(lr_model.regParam, [0.01, 0.1, 1])\
    .build()

# Execute CrossValidator with the parameter grid
crossval = CrossValidator(estimator=cv_lr_model,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="DEP_DELAY_NEW", predictionCol="prediction", metricName="rmse"),
                          numFolds=5)

# Train the tuned model and establish our best model
cvModel = crossval.fit(train)
best_cvModel = cvModel.bestModel

# Return the summary of the best model
# cv_lr_summary = best_cvModel.summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate

# COMMAND ----------

lr_model.summary.meanAbsoluteError

# COMMAND ----------

lr_model.summary.rootMeanSquaredError

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
glr_pred = glr_model.transform(valid)
cv_pred = best_cvModel.transform(valid)

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="DEP_DELAY_NEW", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lr_pred)
glr_rmse = evaluator.evaluate(glr_pred)
cv_rmse = evaluator.evaluate(cv_pred)
print("Root Mean Squared Error (RMSE) on test data: /n")
print(f'Linear Regression: {lr_rmse}')
print(f'Generalized Linear Regression: {glr_rmse}')
print(f'Cross Validation: {cv_rmse}')

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="DEP_DELAY_NEW", predictionCol="prediction", metricName="mae")
lr_mae = evaluator.evaluate(lr_pred)
glr_mae = evaluator.evaluate(glr_pred)
cv_mae = evaluator.evaluate(cv_pred)
print("Mean Average Error (MAE) on test data: /n")
print(f'Linear Regression: {lr_mae}')
print(f'Generalized Linear Regression: {glr_mae}')
print(f'Cross Validation: {cv_mae}')

# COMMAND ----------


########## Geenie Generated!! #####################

def calculate_rmsle(predictions: pred, trueLabel: str, predictionCol: str) -> float:
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    predictions (DataFrame): The DataFrame containing the true labels and predictions.
    trueLabel (str): The name of the column containing the true labels.
    predictionCol (str): The name of the column containing the predictions.

    Returns:
    float: The RMSLE value.
    """
    # Add 1 to both predictions and labels to avoid log(0)
    df = predictions.withColumn("log_true_plus_1", log(col(trueLabel) + 1))\
                    .withColumn("log_pred_plus_1", log(col(predictionCol) + 1))
    
    # Calculate the squared log error
    df = df.withColumn("squared_log_error", (col("log_true_plus_1") - col("log_pred_plus_1")) ** 2)
    
    # Calculate the mean squared log error
    msle = df.select(avg("squared_log_error")).first()[0]
    
    # Return the square root of MSLE as RMSLE
    return sqrt(msle)

# Assuming 'pred' is your DataFrame containing the predictions and true labels
rmsle_value = calculate_rmsle(pred, "DEP_DELAY_NEW", "prediction")
print(f"Root Mean Squared Logarithmic Error (RMSLE) on test data = {rmsle_value}")

# COMMAND ----------

# add RMSLE
# Include F3 evaluation 
# What is the RMSE / MAE / RMSE of your baseline (mean departure times)?


# COMMAND ----------

### Feature Importance

# Extract feature importances (coefficients)
lr_coefficients = lr_model.coefficients.toArray()
glr_coefficients = glr_model.coefficients.toArray()
cv_coefficients = best_cvModel.coefficients.toArray()
feature_importances = list(zip(feature_list, lr_coefficients,glr_coefficients,cv_coefficients))

# Sort the list by importance in descending order
lr_sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
glr_sorted_feature_importances = sorted(feature_importances, key=lambda x: x[2], reverse=True)
cv_sorted_feature_importances = sorted(feature_importances, key=lambda x: x[3], reverse=True)

# Select the top 10 features
lr_top_10_features = lr_sorted_feature_importances[:10]
glr_top_10_features = glr_sorted_feature_importances[:10]
cv_top_10_features = cv_sorted_feature_importances[:10]

# COMMAND ----------

len(feature_list)

# COMMAND ----------

lr_top_10_features[6]

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
plot_top_features(glr_top_10_features, 2, 'Top 10 Features for Generalized Linear Regression Model')
plot_top_features(cv_top_10_features, 3, 'Top 10 Features for Cross-Validated Model')

# COMMAND ----------

# Select the top 10 features
lr_bottom_10_features = lr_sorted_feature_importances[-10:]
glr_bottom_10_features = glr_sorted_feature_importances[-10:]
cv_bottom_10_features = cv_sorted_feature_importances[-10:]

# Plotting top 10 features for each model
plot_top_features(lr_bottom_10_features, 1,'Bottom 10 Features for Linear Regression Model')
plot_top_features(glr_bottom_10_features, 2, 'Bottom 10 Features for Generalized Linear Regression Model')
plot_top_features(cv_bottom_10_features, 3, 'Bottom 10 Features for Cross-Validated Model')

# COMMAND ----------

# Print the coefficients and intercept for generalized linear regression model
# print("Coefficients: " + str(model.coefficients))
# print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
# summary = model.summary
# print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
# print("T Values: " + str(summary.tValues))
# print("P Values: " + str(summary.pValues))
# print("Dispersion: " + str(summary.dispersion))
# print("Null Deviance: " + str(summary.nullDeviance))
# print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
# print("Deviance: " + str(summary.deviance))
# print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
# print("AIC: " + str(summary.aic))
# print("Deviance Residuals: ")
# summary.residuals()

# COMMAND ----------

lr_model.summary

# COMMAND ----------

lr_pred.count()

# COMMAND ----------

lr_model.summary.residuals.count()

# COMMAND ----------

