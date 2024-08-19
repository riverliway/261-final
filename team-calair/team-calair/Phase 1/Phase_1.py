# Databricks notebook source
# MAGIC %md
# MAGIC ##CalAir: Predicting Domestic Flight Delays Using Machine Learning
# MAGIC A project by the CalAir Data Dcience Team
# MAGIC
# MAGIC ## Phase Leader Plan
# MAGIC
# MAGIC | Profile Picture | Phase Leader | Name | Email |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | <img src="https://ca.slack-edge.com/T0WA5NWKG-U0486G97WHH-9414c9d22a6a-512" alt="Nick pfp" width="200"/> | Phase 1 Leader | Nick Luong | nicholaskluong@berkeley.edu |
# MAGIC | <img src="https://ca.slack-edge.com/T0WA5NWKG-U04HXPPK3U2-62453f7b0cf9-512" alt="Abdul pfp" width="200"/> | Phase 2 Leader | Abdul Alfozan | aalfozan@berkeley.edu |
# MAGIC | <img src="https://ca.slack-edge.com/T0WA5NWKG-U03RH929XPB-ccbd87109eb0-512" alt="Darby pfp" width="200"/> | Phase 3 Leader | Darby Brown |  darbybrown@berkeley.edu |
# MAGIC | <img src="https://ca.slack-edge.com/TS92TKBGC-U035EKNF8LD-93dd7de605f6-512" alt="River pfp" width="200"/> | Phase 4 (HW 5) Leader | River Schieberl | riverli@berkeley.edu |
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Project Abstract
# MAGIC
# MAGIC Airline delays represents one of the largest cost factors for airports and is one of the most common painpoints for flyers around the world. The objective of this project is to predict flight delays for US domestic flights, focusing on the delay duration in minutes. We are provided with historical flight and weather data (2015 - 2021) regarding flights and weather from the Department of Transportation and National Oceanic and Atmospheric Administration with the goal to develop a model that can predict the length of a delay two hours before the scheduled departure time. This prediction will help airlines, airports, and passengers to better manage time and resources. We will explore regression approaches and evaluate multiple machine learning algorithms, including Ordinary Least Squares (OLS) Linear Regression, XGBoost, and Multi-Layer Perceptron (MLP). We will use Root Mean Squared Log Error (RMSLE) and MSE as the metrics to assess model performance.
# MAGIC
# MAGIC We have decided to treat the problem as a regression problem, focusing on predicting the delay duration in minutes, which is a continuous outcome. We will leverage evaluation metrics standard to regression tasks (listed above), and translate predictions to a binary delay/no delay outcome for further supplemental evaluation using the F3 score classification metric.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Gantt Chart: Project Phases and Work Packages
# MAGIC
# MAGIC Development of an ML pipeline will require many tasks and interdependencies. Below is a Gantt chart that details our high level execution plan and key milestones. For a more detailed look at work packages and tasks, view the full Gantt chart [here.](https://docs.google.com/spreadsheets/d/1wVPY6MclSDAu62DsckMr3jBGuwGNDblA4pBhu7xhxHk/edit?usp=sharing)
# MAGIC
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/Redo_Gantt_CalAir_261.png?raw=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Data Description
# MAGIC
# MAGIC **Datasets:**
# MAGIC
# MAGIC We will use the following datasets to inform our analysis. These data have been joined for direct comparison. We will use this joined dataset for efficiency, conducting the join by hand and incorporating later data (2022-2023) if time permits.
# MAGIC
# MAGIC - **Flights Data:**
# MAGIC   - Source: [U.S. Department of Transportation (DOT) Bureau of Transportation Statistics](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)
# MAGIC   - Dimensions: 31,746,841 x 109 (for 2015-2019)
# MAGIC   - Subsets: Q1 2015, Q1+Q2 2015, 2019, 2015-2021
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`
# MAGIC   - Summary: Sourced from the Bureau of Transportation Statistics, this dataset includes a quantitative summary of flights' departure performance, arrival performance, cancellations and diversions. Additional features represent the carrier, dates/times, origin/destination information, and reason for delay. The data has been parsed into small subsets which will be used for development. Initial model pipelines will be developed with 3-month dataset from Q1 2015 and, susequently, on a 12-month dataset from 2019. A best-performing model pipeline will be trained and tested on the complete dataset (2015-2021).
# MAGIC
# MAGIC - **Weather Data:**
# MAGIC   - Source: [National Oceanic and Atmospheric Administration (NOAA)](https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf)
# MAGIC   - Dimensions: 630,904,436 x 177 (for 2015-2019)
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/`
# MAGIC   - Summary: National weather data is included to supplement the flights delay data for 2015-2021, as a significant proportion of delays may be attributed to weather. This dataset includes daily and monthly summary statistics as well as hourly data for precipitation, visibility, wind speed, etc. 
# MAGIC
# MAGIC - **Airport Metadata:**
# MAGIC   - Source: US Department of Transportation (DoT)
# MAGIC   - Dimensions: 18,097 x 10
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data`
# MAGIC   - Summary: This dataset provides relevant metadata about each airport.
# MAGIC
# MAGIC - **Airport Codes:**
# MAGIC   - Source: [DataHub](https://datahub.io/core/airport-codes)
# MAGIC   - Summary: This table contains IATA or ICAO airport codes that will be joined to the flights dataset to represent each airport properly.
# MAGIC
# MAGIC
# MAGIC  We conducted initial analysis of all data to eliminate columns with high proportion of missing data and highly correlated parameters. As a result of this review, we will focus on further analysis of the parameters listed below. These parameters were selected if they (1) had less than 10% missing data and (2) made intuitive sense as possibly related to airport delays. 
# MAGIC  
# MAGIC  Attention will be focused on modeling delay departure times, with additional consideration given to modeling arrival delay times (which must take into account departure delay and flight duration).
# MAGIC
# MAGIC **Parameters**
# MAGIC | Column Name | Description |
# MAGIC |----|----|
# MAGIC | DAY_OF_MONTH | Day of Month |
# MAGIC | DAY_OF_WEEK | Day of Week |
# MAGIC | MONTH | Month |
# MAGIC | ORIGIN | Origin Airport |
# MAGIC | DEST | Destination Airport |
# MAGIC | CRS_DEP_TIME | Scheduled departure time (local time hh:mm). |
# MAGIC | CRS_ARR_TIME | Scheduled arrival time (local time hh:mm). |
# MAGIC | CANCELLED | Cancelled Flight Indicator (1=Yes). |
# MAGIC | DIVERTED | Diverted Flight Indicator (1=Yes). |
# MAGIC | DISTANCE | Distance between airports (miles). |
# MAGIC | (CARRIER/ WEATHER/ NAS/ SECURITY/ LATE_AIRCRAFT)_DELAY | These 5 columns show how much the delays were traceable to different sources. Not all delay is always accounted for. |
# MAGIC | origin_airport_lat | Latitude of Origin Airport |
# MAGIC | origin_airport_lon | Longitude of Origin Airport |
# MAGIC | dest_airport_lat | Latitude of Destination Airport |
# MAGIC | dest_airport_lon | Longitude of Destination Airport |
# MAGIC | HourlyDryBulbTemperature | Hourly dry-bulb temperature, which is commonly used as the standard air temperature reported. |
# MAGIC | HourlyPrecipitation | Amount of precipitation in inches to hundredths over the past hour. A 'T' indicates trace amounts of precipitation. |
# MAGIC | HourlyRelativeHumidity | The hourly relative humidity given to the nearest whole percentage. |
# MAGIC | HourlyStationPressure | Hourly atmospheric pressure observed at the station during the time of observation. Given in inches of Mercury (in Hg). |
# MAGIC | HourlyVisibility | Hourly visibility. This is measured as the horizontal distance an object can be seen and identified given in whole miles. Note visibilities less than 3 miles are usually given in smaller increments (e.g. 2.5). |
# MAGIC | HourlyWindSpeed | Speed of the wind at the time of observation (hourly) given in miles per hour (mph). |
# MAGIC | DEP_DELAY_NEW **(Y variable)**| Number of minutes between scheduled time and actual time. Null if flight was canceled. 0 if on time or early. |
# MAGIC | ARR_DELAY_NEW **(Y variable)**| Number of minutes between scheduled time and actual time. Null if flight was canceled OR diverted. 0 if on time or early. |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Algorithms and Metrics
# MAGIC
# MAGIC
# MAGIC After EDA to inform more intentional feature selection and possible feature engineering, we will implement a selection of ML algorithms of increasing complexity. The goal is to reach a satisfactory model with the least complex model possible in order to maintain model explainability. The following models will be tested:
# MAGIC
# MAGIC 1. **Ordinary Least Squares (OLS)**
# MAGIC    - **Implementation:** Scikit-learn (`sklearn.linear_model.LinearRegression`)
# MAGIC    - Explanation: OLS is a type of linear regression that estimates the relationship between independent variables and the dependent variable by minimizing the sum of the squared differences between the observed and predicted values.
# MAGIC
# MAGIC
# MAGIC 2. **XGBoost**
# MAGIC    - **Implementation:** XGBoost (`xgboost.XGBRegressor`)
# MAGIC    - XGBoost is a powerful gradient boosting algorithm that builds an ensemble of decision trees for regression tasks. Each tree is trained to correct the errors of the previous ones, improving the model's overall accuracy.
# MAGIC
# MAGIC 3. **Multi-Layer Perceptron (MLP)**
# MAGIC    - **Implementation:** Scikit-learn (`sklearn.neural_network.MLPRegressor`)
# MAGIC    - Explanation: MLP is a type of artificial neural network consisting of multiple layers of nodes. Each node, or neuron, applies a weighted sum of its inputs and passes the result through a non-linear activation function. MLPs are capable of capturing complex relationships in the data.
# MAGIC
# MAGIC
# MAGIC **Loss Function:** Root Mean Squared Log Error (RMSLE)
# MAGIC
# MAGIC #### Description of Metrics
# MAGIC - **Root Mean Squared Log Error (RMSLE)**
# MAGIC    - Used for evaluating regression models. Using RMSLE is useful in that it penalizes underestimations far more than overestimations. In the context of predicting flight delays: for a true delay time of 1 hour, we would penalize the model more for predicting a 15 min delay than we would for predicting a 2 hour delay.
# MAGIC    - Formula:
# MAGIC      \\(
# MAGIC      \text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}
# MAGIC      \\)
# MAGIC
# MAGIC - Secondery metric:  **Mean Squared Error (MSE)**
# MAGIC    - Used for evaluating regression models.
# MAGIC    - Formula:
# MAGIC      \\(
# MAGIC      \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
# MAGIC      \\)
# MAGIC
# MAGIC
# MAGIC **Note:** For a regression task, a prediction of -10 mins delay will be treated as a prediction of on-time (0 mins delay).
# MAGIC
# MAGIC #### Secondary Evaluation Method: Classification Metrics 
# MAGIC
# MAGIC After model training and test, we will convert results to a binary format to evaluate select classification metrics (F-Score) based on the 15-minute delay threshold.
# MAGIC
# MAGIC **F3-Score**
# MAGIC    - The F3 Score emphasizes recall more than both the F1 and F2 Scores, reducing the likelihood of false negatives (predicting no delay when there is a delay).
# MAGIC    - Formula:
# MAGIC
# MAGIC      \\(
# MAGIC       F3 = \frac{(1 + 3^2) \cdot P \cdot R}{3^2 \cdot P + R} = 10 \cdot \frac{P \cdot R}{9 \cdot P + R}
# MAGIC      \\)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Machine Learning Pipeline
# MAGIC
# MAGIC ![p](https://iot.alfozan.io/ML_pipeline.png)
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Pipeline Stages Description
# MAGIC
# MAGIC 1. **EDA & Data Ingestion**
# MAGIC    - **Description:** This stage involves loading flight and weather data, and conducting Exploratory Data Analysis (EDA) to understand data structure, identify patterns, detect anomalies, and summarize characteristics. Key activities include importing datasets from Azure Blob Storage, understanding data distribution and relationships, identifying missing values and outliers, and visualizing data (e.g., delays over time, weather impact).
# MAGIC
# MAGIC 2. **Data Cleaning**
# MAGIC    - **Description:** Preparing the data for analysis by addressing issues identified during EDA to ensure accuracy and consistency. Key activities include handling missing data, removing or correcting outliers, standardizing and normalizing data, and encoding categorical variables (e.g., airport codes, airlines).
# MAGIC
# MAGIC 3. **Feature Engineering**
# MAGIC    - **Description:** Creating new features from existing data by transforming raw data into meaningful inputs. Key activities include creating new features, transforming existing features (e.g., scaling delays), generating interaction features, and reducing dimensionality (i.e. PCA) if necessary.
# MAGIC
# MAGIC 4. **Data Splitting**
# MAGIC    - **Description:** Splitting the prepared data into training, validation, and testing sets to assess model generalization performance.
# MAGIC
# MAGIC 5. **Model Training**
# MAGIC    - **Description:** Applying machine learning algorithms to the training data to build predictive models
# MAGIC
# MAGIC 6. **Model Evaluation**
# MAGIC    - **Description:** Assessing trained models on the validation dataset to ensure they generalize well to unseen data. Key activities include evaluating models using metrics (e.g. RMSLE, MSE, F3), comparing performance, analyzing errors, and visualizing performance.
# MAGIC
# MAGIC 7. **Model Tuning**
# MAGIC    - **Description:** Fine-tuning the models to achieve optimal performance by iteratively adjusting parameters based on evaluation feedback. Key activities include tuning hyperparameters, re-evaluating the models, and ensuring optimal performance criteria are met.
# MAGIC
# MAGIC 8. **Model Selection**
# MAGIC    - **Description:** Selecting the best-performing model based on evaluation metrics and preparing it for deployment
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Preliminary EDA
# MAGIC
# MAGIC In Phase 1, we did an initial exploratory analysis to familiarize ourselves with the data we plan to utilize. Below are some of our initial observations as we continue to explore different EDA techniques to get a stronger understanding
# MAGIC
# MAGIC #### Summary Statistics
# MAGIC | Column | Mean | Std | Min | 25% | 50% | 75% | Max |
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | ARR_DELAY_NEW | 33.2 | 52.3 | 1.0 | 6.0 | 16.0 | 38.0 | 1971.0 |
# MAGIC | DEP_DELAY_NEW | 32.6 | 52.4 | 1.0 | 5.0 | 15.0 | 38.0 | 1988.0 |
# MAGIC | WEATHER_DELAY | 45.5 | 72.7 | 1.0 | 10.0 | 22.0 | 51.0 | 1152.0 |
# MAGIC | CARRIER_DELAY | 33.9 | 58.7 | 1.0 | 8.0 | 17.0 | 36.0 | 1971.0 |
# MAGIC
# MAGIC ### Correlation Matrix
# MAGIC <img src = 'https://nickkluong.github.io/w261_final_project/Correlation_Matrix.png'>
# MAGIC
# MAGIC **Preleminary observations based on EDA:** Arrival and Departure delays seem to be similar in terms of severity and are highly correlated (coeff. 0.96). Based on high correlation of these two outcomes, we expect a model of arrival delays to be a relatively simple linear model based on departure delay and flight distance. Weather delays tend to be longer and have higher variance than other delays.
# MAGIC
# MAGIC ### Other Observations
# MAGIC <img src = 'https://nickkluong.github.io/w261_final_project/arr_time_by_airline.png'>
# MAGIC
# MAGIC Frontier Airlines has significantly more delays than any other carrier. Budget travelers... beware!
# MAGIC
# MAGIC <img src = 'https://nickkluong.github.io/w261_final_project/avg_delay_by_airport.png'>
# MAGIC <img src = 'https://nickkluong.github.io/w261_final_project/dep_delay_by_airport.png'>
# MAGIC
# MAGIC In the above graphs, we see that Trenton-Mercer Airport (TTN) Airport struggles with a higher average delays for both departures and arrival flights than any other airports. This issue demonstrates logistical challenges, outside of airlines' control, that these airports are facing such as limited runway capacity, high traffic volume, or inadequate ground services.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment Breakdown
# MAGIC
# MAGIC High-level estimations. For detailed task-breakdown and ownership, refer to [Gantt Chart](https://docs.google.com/spreadsheets/d/1wVPY6MclSDAu62DsckMr3jBGuwGNDblA4pBhu7xhxHk/edit?usp=sharing).
# MAGIC
# MAGIC #### Phase 1
# MAGIC | Phase | Engineer | Description | Time Estimation |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | 1 | River | Create proposal outline | 1 hour |
# MAGIC | 1 | Abdul | Create team cloud storage | 2 hours |
# MAGIC | 1 | Nick | Abstract | 1 hour |
# MAGIC | 1 | Darby | Gantt Chart | 2 hour |
# MAGIC | 1 | Darby | Data Description | 2 hour |
# MAGIC | 1 | Abdul | ML Algorithms and Metrics | 2 hour |
# MAGIC | 1 | Abdul | Block diagram of ML Pipeline | 1 hour |
# MAGIC | 1 | River + Nick | Initial EDA | 4 hours |
# MAGIC | 1 | Nick | Tasks table 'Credit Assignment Plan' | 1 hour |
# MAGIC
# MAGIC #### Expected Phase 2 and 3
# MAGIC | Phase | Engineer | Description | Time Estimation |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | 2 | River + Nick | In-Depth EDA | 10 hour |
# MAGIC | 2 | Darby + Abdul | Feature Engineering and Data Cleaning | 10 hour |
# MAGIC | 2 | Everyone | Linear Regression Pipeline | 10 hour |
# MAGIC | 2 | Everyone | Assembling the deliverable | 3 hour |
# MAGIC | 3 | Everyone | XGB and MLP Modeling | 20 hour |
# MAGIC | 3 | Everyone | Final ML Pipeline| 10 hour |
# MAGIC | 3 | Everyone | Assembling the deliverable | 3 hour |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We're looking forward to tackle the issue of airline delays for the next couple weeks. To summarize, we established our objective and defined a game plan that we plan to execute. In the next phases, we'll proceed with exploratory data analysis (EDA), engage in data pre-processing and feature engineering, and implement our models. We hope to drive meaningful insights and solutions and produce a deliverable that can provides a clear recommendation to the problem that is airline delays
# MAGIC