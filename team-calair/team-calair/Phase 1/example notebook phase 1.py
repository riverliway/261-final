# Databricks notebook source
# MAGIC %md
# MAGIC #Predicting Flight Delays for US Air-Travelling Passengers
# MAGIC #__w261 Final Project - Phase 1 - Team 4-3__

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Team Details
# MAGIC
# MAGIC | Name | Email | Photo | Location|
# MAGIC | --- | --- | --- | ---|
# MAGIC | Aashray Puri | aashraypuri@berkeley.edu | <img src="https://ca.slack-edge.com/T0WA5NWKG-U04ETQ30MPT-188c0f4aae3f-512" alt="Aashray Puri" width="300" height="300"> | Houston, TX |
# MAGIC | Adam Hyman | adamhyman@ischool.berkeley.edu | <img src="https://ca.slack-edge.com/T0WA5NWKG-U029P0HJN4V-d7e03f783007-512" alt="Adam Hyman" width="300" height="300"> | Los Angeles, CA |
# MAGIC | Anna Li | anna.li2@ischool.berkeley.edu | <img src="https://ca.slack-edge.com/T0WA5NWKG-U04CVLCCY93-140cdb66b4d6-512" alt="Anna Li" width="300" height="300"> | Seattle, WA |
# MAGIC | Spencer Hodapp | shodapp@ischool.berkeley.edu | <img src="https://ca.slack-edge.com/T0WA5NWKG-U034464DESH-4f960f3453da-512" alt="Spencer Hodapp" width="300" height="300"> | Washington, DC |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Phase Leader Plan
# MAGIC
# MAGIC | Week | Phase | Leader | Backup/Co-Leader |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 1 | 1 | Aashray | Adam |
# MAGIC | 2 | 2 | Adam | Anna |
# MAGIC | 3 | 2 | Anna | Hodapp |
# MAGIC | 4 | 3| Hodapp | Aashray |
# MAGIC | 5 | 3| Aashray | Adam | 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Project Abstract
# MAGIC
# MAGIC We are a group of Data Scientists who have been approached for a project by a committee focused on reducing stress for air-travelling passengers in the US. In particular, the committee is hoping to predict flight departure delays for US air-travelling passengers so as to reduce economic losses as well as inconvenience for the passengers. The committee has furnished 3 datasets (namely these are the Stations, Weather and Flights datasets) for our use, but have left the approach of the project open to us. A description of the data and some elementary data exploration notes are provided later in this report. 
# MAGIC
# MAGIC In this project, our group aims to use modern Machine Learning approaches to help the committee look out for US air-travelling passengers. We will first perform Exploratory Data Analysis (EDA) to gain a better understanding of our data, and will use Data Cleaning/Pre-processing techniques to prepare out datasets for Modeling and analysis. Our approach to predicting flight delays will be from a classification perspective, where our baseline model (i.e the one we will try to out-perform) will be a simple logistic regreesion. What's more, we will be experimenting with boosting, bagging, deep learning and ensemble models to try and predict flight departure delays in a better way than our baseline. After training each model, we will evaluate it using F1 score, as it balances precision and recall. 
# MAGIC
# MAGIC Upon completion of the project, we hope to select a model that will aid in improving the experience of air-traveling Americans by accurately predicting flight delays.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Gantt Chart of Project Phases & Sub-Tasks
# MAGIC
# MAGIC Below is a visual summary of the 3 phases which make-up this project, and the various sub-tasks which are included in each phase:
# MAGIC
# MAGIC
# MAGIC <img src ='https://adamhyman-public.s3.us-west-2.amazonaws.com/W261/gantt.png'>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Description 
# MAGIC
# MAGIC We have been provided 3 datasets for this project - the Stations, Weather and Flights datasets. Below is a brief description of the data from each of the Datasets:
# MAGIC
# MAGIC - The Stations dataset provides meta-data on each airport in the US. This data was pulled from the US Department of Transportation, and there are 12 columns of data provided. Initial EDA has suggested that all 12 of these columns have less than 10% null values.
# MAGIC
# MAGIC - The Weather dataset originates from the National Oceanic and Atmospheric Administration and includes data from 2015 through 2021. The data includes 127 columns for further analysis, with a mix of numerical and categorical variable types. Initial EDA has suggested that 111 of these 127 columns have more than 10% null values.
# MAGIC
# MAGIC - The Flights dataset comes from the US Department of Transportation (DOT), and also includes data from 2015 through 2021. The dataset has 109 columns for further analysis, with a mix of numerical and categorical variable types. Initial EDA has suggested that 53 of these 109 columns have more than 10% null values.
# MAGIC
# MAGIC
# MAGIC | Dataset | Total Num of Cols | Cols with < 10% Null Values| Names of Cols with < 10% Null Values|
# MAGIC | --- | ---| ---| --- |
# MAGIC | Stations | 12 | 12| 'usaf', 'wban', 'station_id', 'lat', 'lon', 'neighbor_id', 'neighbor_name', 'neighbor_state', 'neighbor_call', 'neighbor_lat', 'neighbor_lon', 'distance_to_neighbor'|
# MAGIC | Weather | 127 | 16 |'HourlyRelativeHumidity', 'HourlyDewPointTemperature', 'REM', 'HourlyDryBulbTemperature', 'HourlyWindDirection', 'ELEVATION', 'LONGITUDE', 'LATITUDE', 'NAME', 'YEAR', 'STATION', 'DATE', 'HourlyWindSpeed', 'SOURCE', 'REPORT_TYPE', 'HourlyWindSpeedDouble'|
# MAGIC | Flights | 109| 56| 'AIR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'ARR_DELAY_GROUP', 'ARR_DEL15', 'ARR_DELAY_NEW', 'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'WHEELS_OFF', 'TAXI_OUT', 'DEP_DELAY_GROUP', 'DEP_DEL15', 'DEP_DELAY_NEW', 'DEP_DELAY', 'DEP_TIME', 'TAIL_NUM', 'CRS_ELAPSED_TIME', 'QUARTER', 'DISTANCE_GROUP', 'DIV_AIRPORT_LANDINGS', 'MONTH', 'ORIGIN_STATE_NM', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_ABR', 'ORIGIN_CITY_NAME', 'ORIGIN', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_AIRPORT_ID', 'OP_CARRIER_FL_NUM', 'OP_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_UNIQUE_CARRIER', 'FL_DATE', 'DAY_OF_WEEK', 'DAY_OF_MONTH', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEP_TIME_BLK', 'DISTANCE', 'FLIGHTS', 'DIVERTED', 'CANCELLED', 'ARR_TIME_BLK', 'CRS_ARR_TIME', 'CRS_DEP_TIME', 'DEST_CITY_MARKET_ID', 'DEST_WAC', 'DEST_STATE_NM', 'DEST_STATE_FIPS', 'DEST_STATE_ABR', 'DEST_CITY_NAME', 'DEST', 'YEAR'|
# MAGIC
# MAGIC The weather and flights data has been provided in different time intervals (eg. Weather & Flights have been provided for 3 month, 6 month, 1 year & 6 year time intervals). Moreover, the flight and weather datasets have also been provided seperately in a combined format (OTPW). We plan to utilize this joined dataset unless we have extra time to merge the datasets ourselves. We are also looking to supplement the provided datasets with data relating to holidays and airline alliances. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Machine Learning Algorithms and Metrics 
# MAGIC
# MAGIC To begin with we are approaching this project from a classification perspective, and so we decided to keep our baseline model as a simple Logisitic Regression model, which only takes numerical features as an input. The idea behind this is to try and see if the use of both numerical and categorical features, as well as more sophisticated modeling techniques can help improve on the baseline model. As such we have selected five distinct algorithms (listed in table format below), each employing a unique approach. Our evaluation process will determine which technique yields the optimal results. Subsequently, we will construct an ensemble model that leverages the strengths of all these diverse models.
# MAGIC
# MAGIC For the loss function, we have opted for binary cross-entropy, which penalizes incorrect classifications more severely when the model's confidence is high. Lower values of binary cross-entropy indicate better performance. Our chosen evaluation metric is the F1 score, which combines precision and recall to assess the classification models' performance.
# MAGIC
# MAGIC ##### Algorithms to be used:
# MAGIC
# MAGIC | Algorithm Number | Algorithm Type | Algorithm Name | Implementation Details | Loss Function | Evaluation Metric | Rationale for using this Model|
# MAGIC | --- | --- | --- | --- | --- | --- | --- |
# MAGIC | Baseline | Binary Classification | Logistic Regression | Only using Numerical Features | Binary Cross Entropy Loss | F1 Score | Simple and easy to interpret |
# MAGIC | 1 | Binary Classification | Logistic Regression | Using both Numerical & Categorical Features| Binary Cross Entropy Loss| F1 Score | Same as baseline, but also using categorical features  |
# MAGIC | 2 | Boosting | XGBoost | Using both Numerical & Categorical Features | Binary Cross Entropy Loss | F1 Score |Handles Missing data easily. Includes L1/L2 regularization. Easily Scalable|
# MAGIC | 3 | Bagging | Random Forest/ Decision Tree | Using both Numerical & Categorical Features | Binary Cross Entropy Loss | F1 Score | Less prone to over-fitting. Works well with numerical as well as categorical features |
# MAGIC | 4 | Deep Learning | Neural Network | Tensorflow/PyTorch on both Numerical & Categorical Features | Binary Cross Entropy Loss | F1 Score | Optimal for exploring non-linear relationships. Easily Scalable |
# MAGIC | 5 | Ensemble | Ensemble of Above Models | Using both Numerical & Categorical Features | Binary Cross Entropy Loss | F1 Score | Take advantage of strengths of each model |
# MAGIC
# MAGIC
# MAGIC where *Binary Cross-Entropy Loss* = \\( -\\frac{1}{n} \\sum_{i=1}^{n} -y_i \\log(\\hat{y_i}) + (1 - y_i) \\log(1 - \\hat{y_i}) \\) and
# MAGIC <br>
# MAGIC *Modified F1 Score* = \\(\\frac{ 2 \\cdot precision \\cdot recall}{precision + recall} = \\frac{TP}{TP + \\frac{1}{2} (FP + FN)} \\)
# MAGIC %md
# MAGIC
# MAGIC #### Machine Learning Pipeline
# MAGIC
# MAGIC <img src ='https://adamhyman-public.s3.us-west-2.amazonaws.com/W261/mlpipeline.png'>
# MAGIC
# MAGIC
# MAGIC Our preliminary plan is to begin with Exploratory Data Analysis (EDA) and Data Cleaning to become more familiar with the dataset(s). Up until this point, we've conducted very simple EDA to size up the dataset but our goal is to go deeper to understand:
# MAGIC
# MAGIC - Descriptive Statistics of the Datasets
# MAGIC - Whether our Dataset already includes a reliable Target Variable, or if we need to create our own Target Variable
# MAGIC - The balance of data in terms of data types (i.e numerical features vs categorical features)
# MAGIC - Which of the 200+ variables we have access to, are best suited for including in our Models
# MAGIC - etc.
# MAGIC
# MAGIC EDA and Data Cleaning tends to go hand in hand with Data Preparation and Feature Engineering, and so upon understanding the data at hand better, we will start selecting the relevant features for our Models, and Feature Engineer them accordingly. For instance, numerical features will be z-score standardized and categorical features will be converted to one-hot encodings. We expect these steps of EDA, Data Cleaning, Data Preparation and Feature Engineering to take up the bulk of the time spent on this project, and these steps are by far the most important to making sure our Models are appropriate/accurate. After completing the Feature Engineering, we hope to start implementing and configuring our Baseline Model (a Logistic Regression model). We believe that the steps of Model Training, Model Optimization and Model Evaluation will also be inter-connected steps, ones where we will configure our models a certain way, see how they perform, and then adjust the relevant parameters to see if our models can be made sharper. The goal here is to make sure we learn generalizable patterns from the training data, but do not overfit the models to the training data. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Preliminary EDA
# MAGIC
# MAGIC In Phase 1, we wanted to do some exploratory EDA to gain a better understanding of the data. In particular, we looked at the correlation between the non-null numerical features for three months of airline data. In the upcoming phases, we will utilize correlation plots and other EDA techniques to continue exploring the data and to perform feature selection.
# MAGIC
# MAGIC <img src = 'https://adamhyman-public.s3.us-west-2.amazonaws.com/W261/corr.png'>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Credit Assignment Plan
# MAGIC
# MAGIC | Phase | Task | Contributers | Contributions (hours) |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 1 | Set up Databricks cluster and Azure blob storage| Aashray, Adam | 1.5 |
# MAGIC | 1 | Create Gantt diagram of project timeline| Aashray | 1 |
# MAGIC | 1 | Write project abstract | Aashray, Adam, Anna, Hodapp | 0.5 |
# MAGIC | 1 | Discuss machine learning algorithms | Aashray, Adam, Anna, Hodapp | 2 |
# MAGIC | 1 | Data Description | Aashray, Adam, Anna, Hodapp | 1 |
# MAGIC | 1 | Basic EDA | Anna | 1 |
# MAGIC | 1 | Pipeline Steps | Aashray, Adam, Anna, Hodapp | 1 |
# MAGIC | 1 | Delegation of Steps | Aashray, Adam, Anna, Hodapp | 2 |
# MAGIC | 1 | Write Conclusion and Issues sections | Aashray, Adam, Anna, Hodapp | 0.5 |
# MAGIC
# MAGIC ##### Expected Credit Assigment Plan Phase 2 & 3
# MAGIC
# MAGIC | Phase | Task | Contributers | Contributions (hours) |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 2 | Making adjustments to Project Plan after Phase 1 Feedback | Aashray, Adam, Anna, Hodapp | 1 |
# MAGIC | 2 | More in-depth EDA | Aashray, Adam | 2 |
# MAGIC | 2 | Feature Engineering | Anna, Hodapp | 2 |
# MAGIC | 2 | Configuring Baseline Model | Aashray, Adam, Anna, Hodapp | 1 |
# MAGIC | 2 | Initial Configuring of Models 1-5 | Aashray, Adam, Anna, Hodapp | 2 |
# MAGIC | 2 | Working on Deliverable for Phase 2 | Aashray, Adam, Anna, Hodapp | 2-3 |
# MAGIC | 3 | Making adjustments to Project Plan after Phase 2 Feedback | Aashray, Adam, Anna, Hodapp | 1 |
# MAGIC | 3 | Configuring all remaining Models | Aashray, Adam, Anna, Hodapp | 2 |
# MAGIC | 3 | Working on Deliverable for Phase 3 | Aashray, Adam, Anna, Hodapp | 2-3 |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Conclusion and Next Steps
# MAGIC
# MAGIC In Phase 1, we have introduced the team, outlined the problem we are trying to solve, and provided the framework that we will be following for the duration of the project. In the upcoming weeks, we will continue with EDA, data pre-processing / feature engineering, and the implementation of our models. Throughout these steps, we will document our process and our progress in our deliverables and in our credit assignment plans.
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Open Issues / Problems
# MAGIC
# MAGIC An open issue we have is a final determination whether we want to go with a regression or a classification approach. We may try both types of modeling and see which has a better performance. We will attend office hours and talk with our peers to learn more about the pros and cons of each approach. 
# MAGIC