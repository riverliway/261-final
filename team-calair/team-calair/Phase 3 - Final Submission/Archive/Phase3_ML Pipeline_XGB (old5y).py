# Databricks notebook source
# MAGIC %md
# MAGIC ### XGBoost with Time-Series Rolling Cross Validation

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

# get training and test datasets (5y)
df_train = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/train_4y")
df_test = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_5y/test_1y")

# get training and test datasets (1y), for quick testing
# Cleaned_Data_3m = spark.read.parquet(f"{team_blob_url}/Phase_3/Cleaned_Data_1y")
# df_train, df_test = Cleaned_Data_3m.randomSplit([0.8, 0.2], seed=42)

# Display the DataFrame
display(df_train.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Out Data

# COMMAND ----------

import matplotlib.pyplot as plt

departure_delay_pd = df_train.select('DEP_DELAY_NEW').toPandas()
plt.hist(departure_delay_pd, bins=50)

# COMMAND ----------

#engineered capped delay (to eliminate super long tail)
departure_capped_delay_pd = df_train.select('dep_capped_delay').toPandas()
plt.hist(departure_capped_delay_pd, bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Response and Predictor Variables

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# COMMAND ----------

#Define Y
Y = 'dep_capped_delay'

#Define X set
categoricals = ['DAY_OF_WEEK_onehot','Departure_Time_Buckets_onehot', 'OP_CARRIER_onehot', 'DEST_onehot', 'ORIGIN_onehot']
numerics = ['scaled_features']
booleans = ['is_extreme_weather', 'is_holiday', 'is_vert_obscured']

#Make a list of column names to tell us what is in numerics
numerical_cols = ['DISTANCE', 'ELEVATION', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'HoursSinceLastFlight', 'PrevFlightDelay','origin_pagerank', 'dest_pagerank']

#Compile X set
X = categoricals + numerics + booleans
df_train = df_train.select(X + [Y] + ['FL_DATE'])
df_test = df_test.select(X + [Y] + ['FL_DATE'])

# COMMAND ----------


#back out the column names for one-hot since we can't retrieve them from feature engineered dataset

feature_list = []

for column in categoricals:
    metadata = df_train.schema[column].metadata
    metadata = metadata['ml_attr']['attrs']['binary']
    # print(f"Metadata for {column}: {metadata}")
    idx_name = [column + '_' + x['name'] for x in metadata]
    feature_list.extend(idx_name)

feature_list.extend(numerical_cols)
feature_list.extend(booleans)
print(len(feature_list))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up input data for model
# MAGIC
# MAGIC Because of the distrubion of the data, it makes sense to use GLM instead of linear regression framework in Apache Spark. [More](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#linear-regression).

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

#Use VectorAssembler to compile all data into a single vector column
va = VectorAssembler(inputCols = X, outputCol = 'features')
df_train = va.transform(df_train)
df_test = va.transform(df_test)

df_train = df_train.select(['features', 'dep_capped_delay', 'FL_DATE'])
df_test = df_test.select(['features', 'dep_capped_delay', 'FL_DATE'])

print(df_train.count())
print(df_test.count())

# COMMAND ----------

# display(df_train.limit(5))
# display(df_test.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkXGBRegressor Prototype
# MAGIC
# MAGIC Docs: https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
# Docs: https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html

# Initialize the SparkXGBRegressor with parameters directly
# optimized parameters using grid search
xgboost_model = SparkXGBRegressor(
    features_col='features', 
    label_col='dep_capped_delay',
    max_depth=6,
    n_estimators=150,
    reg_lambda=1,
    reg_alpha=0,
    base_score=0.5,
    learning_rate=0.2,
    gamma=0.05,
    scale_pos_weight=2,
    min_child_weight=1.5,
    num_workers=6
)

# Train the model
xgboost_model = xgboost_model.fit(df_train)

# Make predictions
predictions = xgboost_model.transform(df_test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

evaluator_mae = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE) on test data: {mae}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkXGBRegressor model + Hyperparameter tuning (grid search)
# MAGIC
# MAGIC Docs: https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
# Docs: https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html


# Initialize the parameter grid
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.2, 0.3],
    'n_estimators': [150, 200]
}

# Evaluators
evaluator = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="prediction", metricName="mae")

# Perform grid search
best_rmse = float('inf')
best_params = None
best_model = None
best_model_trained = None
results = []
for max_depth in param_grid['max_depth']:
    for learning_rate in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            xgb_model = SparkXGBRegressor(
                features_col='features', 
                label_col='dep_capped_delay',
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                # base_score=0.5,
                # gamma=0.05,
                # scale_pos_weight=2,
                # min_child_weight=1.5,
                num_workers=3,
            )
            
            xgb_model_trained = xgb_model.fit(df_train)

            # Make predictions
            predictions = xgb_model_trained.transform(df_test)
            rmse = evaluator.evaluate(predictions)
            mae = evaluator_mae.evaluate(predictions)

            results.append(f"Params: max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, MAE={mae}, RMSE={rmse}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}
                best_model_trained = xgb_model_trained
                best_model= xgb_model

print("Experiments")
for l in results:
    print("\t", l)

print(f"Best parameters: {best_params}")
print(f"Best RMSE: {best_rmse}")

# Make predictions
predictions = best_model_trained.transform(df_test)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE) on test data: {mae}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time-Series Cross Validation

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

# Define a function for time-series cross-validation on a ROLLING BASIS
def time_series_cv(data, date_col, model, evaluator, metric, folds=5):
    data = data.orderBy(date_col)
    train_ratio = 1/(folds+1)
    total_rows = data.count()
    
    # Add a row number column
    window_spec = Window.orderBy(date_col)
    data_with_row_number = data.withColumn("row_num", row_number().over(window_spec))

    metrics = []
    coefficients = []

    for fold in range(1,folds+1):
        # Define the training and validation split
        train_end = int(fold * total_rows * train_ratio)
        validation_end = train_end + int(total_rows * train_ratio)
        print(f'fold={fold}, train_end={train_end}, validation_end={validation_end}')

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
cv_metrics, cv_coefficients, best_cvModel = time_series_cv(df_train, 'FL_DATE', best_model, evaluator, 'rmse')

# COMMAND ----------

# MAGIC %md
# MAGIC ## F3 score

# COMMAND ----------

from pyspark.sql.functions import lit, mean, broadcast

# Calculate the average delay
avg_delay_df = df_test.select(mean('dep_capped_delay').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df = broadcast(avg_delay_df)

# Join the average delay with the valid DataFrame
test_with_avg = df_test.crossJoin(avg_delay_df)

# Calculate the average delay
avg_delay_df_train = df_train.select(mean('dep_capped_delay').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df_train = broadcast(avg_delay_df_train)

# Join the average delay with the valid DataFrame
train_with_avg = df_train.crossJoin(avg_delay_df_train)

# COMMAND ----------

# Add delay_no_delay column based on the prediction value (0 for delay, 1 for no delay) for classification and f3 scope calculation  

from pyspark.sql.functions import when
from pyspark.sql.types import DoubleType

#baseline augment data for classification
test_with_avg = test_with_avg.withColumn("delay_binary_pred", when(test_with_avg["avg_delay"] > 15, 1).otherwise(0).cast(DoubleType()))
test_with_avg = test_with_avg.withColumn("delay_binary_label", when(test_with_avg["dep_capped_delay"] > 15, 1).otherwise(0).cast(DoubleType()))

#linear model augment data for classification
# lr_pred = lr_pred.withColumn("delay_binary_pred", when(best_model["prediction"] > 15, 1).otherwise(0).cast(DoubleType()))
# lr_pred = best_model.withColumn("delay_binary_label", when(best_model["dep_capped_delay"] > 15, 1).otherwise(0).cast(DoubleType()))

#CV augment data for classification
cv_pred = predictions.withColumn("delay_binary_pred", when(predictions["prediction"] > 15, 1).otherwise(0).cast(DoubleType()))
cv_pred = cv_pred.withColumn("delay_binary_label", when(predictions["dep_capped_delay"] > 15, 1).otherwise(0).cast(DoubleType()))

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluators for precision and recall
precision_evaluator = MulticlassClassificationEvaluator(labelCol="delay_binary_label", predictionCol="delay_binary_pred", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="delay_binary_label", predictionCol="delay_binary_pred", metricName="weightedRecall")

# Calculate baseline precision and recall
baseline_precision = precision_evaluator.evaluate(test_with_avg)
baseline_recall = recall_evaluator.evaluate(test_with_avg)

# Calculate XGP regression precision and recall
lr_precision = precision_evaluator.evaluate(cv_pred)
lr_recall = recall_evaluator.evaluate(cv_pred)

# Calculate Baseline F3 score
beta = 3
f3_score = (1 + beta**2) * (baseline_precision * baseline_recall) / ((beta**2 * baseline_precision) + baseline_recall)
print(f"Baseline F3 Score: {f3_score}")

# Calculate XGP F3 score
beta = 3
f3_score = (1 + beta**2) * (lr_precision * lr_recall) / ((beta**2 * lr_precision) + lr_recall)
print(f"XGP Model F3 Score: {f3_score}")

# COMMAND ----------

# MAGIC %md
# MAGIC old F3 score code

# COMMAND ----------

# Add delay_no_delay column based on the prediction value (0 for delay, 1 for no delay) for classification and f3 scope calculation  

from pyspark.sql.functions import when
from pyspark.sql.types import DoubleType
predictions = predictions.withColumn("delay_no_delay", when(predictions["prediction"] > 15, 1).otherwise(0).cast(DoubleType()))
# display(predictions.limit(10))

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluators for precision and recall
precision_evaluator = MulticlassClassificationEvaluator(labelCol="dep_capped_delay", predictionCol="delay_no_delay", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="dep_capped_delay", predictionCol="delay_no_delay", metricName="weightedRecall")

# Calculate precision and recall
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)

# Calculate F3 score
beta = 3
f3_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print(f"F3 Score: {f3_score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra
# MAGIC --------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Baseline

# COMMAND ----------

from pyspark.sql.functions import lit, mean, broadcast

# Calculate the average delay
avg_delay_df = df_train.select(mean('dep_capped_delay').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df = broadcast(avg_delay_df)

# Join the average delay with the valid DataFrame
valid_with_avg = df_train.crossJoin(avg_delay_df)

# Calculate the average delay
avg_delay_df_train = df_train.select(mean('dep_capped_delay').alias('avg_delay'))

# Broadcast the average delay to ensure it's efficiently joined
avg_delay_df_train = broadcast(avg_delay_df_train)

# Join the average delay with the valid DataFrame
train_with_avg = df_train.crossJoin(avg_delay_df_train)

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="avg_delay", metricName="rmse")
baseline_rmse_valid = evaluator.evaluate(valid_with_avg)
print("Root Mean Squared Error (RMSE) on data: /n")
print(f'Baseline valid: {baseline_rmse_valid}')

baseline_rmse_train = evaluator.evaluate(train_with_avg)
print(f'Baseline train: {baseline_rmse_train}')

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="dep_capped_delay", predictionCol="avg_delay", metricName="mae")
baseline_mae_valid = evaluator.evaluate(valid_with_avg)
print("MAE on  data: /n")
print(f'Baseline valid: {baseline_mae_valid}')

baseline_mae_train = evaluator.evaluate(train_with_avg)
print(f'Baseline train: {baseline_mae_train}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Model Predictions

# COMMAND ----------


########## Geenie Generated!! #####################

def calculate_rmsle(predictions, trueLabel: str, predictionCol: str) -> float:
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

