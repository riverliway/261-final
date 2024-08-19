# Databricks notebook source
# MAGIC %pip install elephas

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
parquet_file_path = f"{team_blob_url}/Phase_3/Cleaned_Data_1y"

# Read the Parquet files into a Spark DataFrame
clean_data = spark.read.parquet(parquet_file_path)

# Display the DataFrame
display(clean_data)

# COMMAND ----------

import tensorflow as tf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions
from tensorflow import sparse
import numpy as np

# COMMAND ----------

def create_gnn_features(df: DataFrame):
    """
    Creates a TIME_CHUNK which is an index of time 2 hours long

    Then creates a graph representing all of the features for every pair of origin-dest airports for each time chunk.

    Output format:
    rdd of tuple(TIME_CHUNK, graph)

    Where the graph is formatted sparsely as:
    array of tuple(tuple(ORIGIN, DEST), feature vector)

    Where the ORIGIN and DEST are their indexes
    """
    df = df.withColumn('TIME_CHUNK', functions.floor((functions.unix_timestamp(df.CRS_DEP_TIME) - 1420070400) / (60 * 60 * 2)))
    max_dest_index = clean_data.agg({"DEST_index": "max"}).collect()[0]['max(DEST_index)']
    max_origin_index = clean_data.agg({"ORIGIN_index": "max"}).collect()[0]['max(ORIGIN_index)']
    num_airports = int(max(max_dest_index, max_origin_index)) + 1

    def create_graph_features(row):
        """
        Creates a sparse graph of airports and their connections.

        Output format:
        array of tuple(tuple(ORIGIN, DEST), feature vector)
        """

        regular_features = row['scaled_features'].toArray()[2:8].tolist()
        features = np.array(regular_features + [row['DEP_DELAY_NEW'] / 60, row['ARR_DELAY_NEW'] / 60])
        return (row['TIME_CHUNK'], [
            ((row['ORIGIN_index'], row['DEST_index']), features)
        ])
    
    def condense_graph_features(graph1, graph2):
        graph3 = []

        for (origin, dest), feature in graph1:
            filtered = [x for x in graph2 if x[0][0] == origin and x[0][1] == dest]
            if len(filtered) == 0:
                # Add in any features that are not in the graph2 but are in graph1
                graph3.append(((origin, dest), feature))
            else:
                # The feature vector exists for both graphs, so average them
                graph3.append(((origin, dest), (feature + filtered[0][1]) / 2))

        # Add in any features that are not in the graph1 but are in graph2
        for (origin, dest), feature in graph2:
            filtered = [x for x in graph1 if x[0][0] == origin and x[0][1] == dest]
            if len(filtered) == 0:
                graph3.append(((origin, dest), feature))

        return graph3

    def tensorify_graph_features(timechunk_graph):
        timechunk, graph = timechunk_graph

        indicies = []
        values = []
        for coord, feature in graph:
            for i in range(feature.shape[0]):
                indicies.append([int(coord[0]), int(coord[1]), i])
                values.append(feature[i])

        sparse_tensor = sparse.SparseTensor(indicies, values, [num_airports, num_airports, 8])

        return (timechunk, sparse_tensor)

    timechunk_graphs = df.rdd.map(create_graph_features) \
        .reduceByKey(condense_graph_features) \
        .map(tensorify_graph_features)

    def create_output(graph):
        """
        Creates a tf dense Tensor of the average delay per airport
        """
        delays = tf.cast(sparse.reduce_sum(graph, 1)[:, 6], dtype=tf.float16)

        return delays / num_airports

    # Create a new RDD one timestep in the future so when we join them, we get (t, t+1) pairs for X,y
    timechunk_graphs_future = timechunk_graphs.map(lambda x: (x[0] + 1, create_output(x[1])))

    X_y = timechunk_graphs.join(timechunk_graphs_future) \
        .map(lambda x: (sparse.to_dense(sparse.reorder(x[1][0])), x[1][1]))

    return X_y

X_y = create_gnn_features(clean_data).cache()

# COMMAND ----------

# Number of X-y pairs to train on
dataset_count = X_y.count()
print(dataset_count)

# COMMAND ----------

# Shape of the input graph
input_shape = X_y.first()[0].shape
print(input_shape)

# COMMAND ----------

# Shape of the output vector
output_shape = X_y.first()[1].shape
print(output_shape)

# COMMAND ----------

# Average number of nonzero results (out of 322)
# X_y.map(lambda x: np.count_nonzero(x[1])).mean()

# COMMAND ----------

# Average number of nonzero features (out of 103,684)
# X_y.map(lambda x: np.count_nonzero(np.sum(x[0] != 0, axis=2))).mean()

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def get_model():
    model = Sequential()
    model.add(layers.Input(shape=input_shape, sparse=False))
    model.add(layers.Conv2D(50, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(50, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(50, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(output_shape[0]))

    model.summary()

    return model

model = get_model()

# COMMAND ----------

from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras import metrics
from tensorflow.keras.optimizers.legacy import Adam
from elephas.spark_model import SparkModel

def sync_dataset(X_y_rdd):
    X_train = []
    y_train = np.zeros((dataset_count, output_shape[0]))
    for i, (x, y) in enumerate(X_y_rdd.collect()):
        X_train.append(sparse.reshape(x, [1, input_shape[0], input_shape[1], input_shape[2]]))
        y_train[i] = y

    X_train = sparse.concat(0, X_train)
    print(X_train.shape)
    print(y_train.shape)

    return X_train, y_train

def run_experiment_sync(model, X_y_rdd):
    X_train, y_train = sync_dataset(X_y_rdd)

    model.compile(
        optimizer=Adam(),
        loss=MeanAbsoluteError(),
        metrics=metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    )

    history = model.fit(X_train, y_train, epochs=2, batch_size=4, verbose=1)
    
    return history

def run_experiment(model, X_y_rdd):
    # X_y_rdd.map(lambda x: (sparse.to_dense(x[0]), x[1]))

    model.compile(
        optimizer=Adam(),
        loss=MeanAbsoluteError(),
        metrics=metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    )

    spark_model = SparkModel(model, frequency='epoch', mode='synchronous')

    history = spark_model.fit(X_y_rdd, epochs=2, batch_size=4, verbose=1)
    
    return history

history = run_experiment(model, X_y)

# COMMAND ----------

def show_history_graphs(history):
    plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(history.history['val_mean_absolute_error'], label = 'val_mean_absolute_error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

show_history_graphs(history)