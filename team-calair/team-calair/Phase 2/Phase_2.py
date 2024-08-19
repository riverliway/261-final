# Databricks notebook source
# MAGIC %md
# MAGIC ##CalAir: Predicting Domestic Flight Delays Using Machine Learning
# MAGIC A project by the CalAir Data Dcience Team
# MAGIC
# MAGIC ## Phase 2 Leader Plan
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
# MAGIC Airline delays represents one of the largest cost factors for airports and is one of the most common painpoints for flyers around the world. The objective of this project is to predict flight delays for US domestic flights, focusing on the delay duration in minutes. We are provided with historical flight and weather data (2015 - 2021) regarding flights and weather from the Department of Transportation and National Oceanic and Atmospheric Administration with the goal to develop a model that can predict the length of a delay two hours before the scheduled departure time. This prediction will help airlines, airports, and passengers to better manage time and resources. In this report, we explore linear regression approaches and evaluate multiple machine learning pipelines with the goal of better undersrtanding what features, data manipulation, and modeling choices can improve a basic linear model the most. For this task we chose standard Root Mean Squared Error (RMSE) as the primary metric to assess model performance. A baseline model that calculates the average delay has an RMSE of 51 minutes. Using feature engineering and data manipulation techniques, we were able to improve upon our baseline, and achieve an RMSE of 18 minutes in best-performing models. Armed with the knowledge of key features, a properly manipulated dataset, and a baseline ML pipeline, our team is ready for even more advanced testing in Phase 3.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Gantt Chart: Project Phases and Work Packages
# MAGIC
# MAGIC Development of an ML pipeline requires many tasks and interdependencies. Linked is a Gantt chart that details our high level execution plan and key milestones.  [here.](https://docs.google.com/spreadsheets/d/1wVPY6MclSDAu62DsckMr3jBGuwGNDblA4pBhu7xhxHk/edit?usp=sharing)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Description
# MAGIC Our primary objective is to develop and evaluate various machine learning models that can accurately predict flight delays for airlines, focusing specifically on the impact of departure times. The ability to forecast delays not only improves customer satisfaction by providing more reliable travel information but also assists airlines in optimizing their operational efficiencies.
# MAGIC
# MAGIC #### Data Description
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
# MAGIC #### Feature Engineering
# MAGIC \
# MAGIC <img src = 'https://nickkluong.github.io/w261_final_project/Feature_engineering_flow.png'>
# MAGIC
# MAGIC In the feature engineering phase, the image above highlights our process to transform the raw data into the final dataset used for our machine learning models. 
# MAGIC
# MAGIC For data cleaning, we first had to assign the proper data types to each column to help us with future transformations down the road. We selected features initially based on those that had less than 10% missing data and, intuitively, were possibly related to airport delays. We also filtered out flights that were cancelled and diverted because we believe these flights are out of the scope of our task.
# MAGIC
# MAGIC Next, we wanted to explore additional features that could help explain the behavior behind flight delays. Below is the list of features alongside our logic for designing them:
# MAGIC | New Feature Name | Logic |
# MAGIC |----|----|
# MAGIC | is_vert_obscured | Based on the hourly sky condition free text, we parsed for the code "VV" which indicated vertical obscurity |
# MAGIC | departure_time_buckets | Categorized CRS_DEP_TIME by Morning, Afternoon, Evening, or Night |
# MAGIC | HoursSinceLastFlight | Based on tail number, we looked at the aircraft's previous arrival time and took the total elapsed time between that and it's next departure time |
# MAGIC | PrevFlightDelay | Based on tail number, we grabbed at the aircraft's previous flight departure delay |
# MAGIC | is_extreme_weather | Using the hourly present weather type free text, we parsed the text for conditions we deemed as extreme weather |
# MAGIC | dep_capped_delay | Given the long tail for departure delay, we generated a new field that would cap the delay by it's 95th percentile |
# MAGIC | is_holiday | Flagged flights if the flight date was on a holiday |
# MAGIC
# MAGIC In addition, we dropped rows where we had empty data because we couldn't infer anything from the missing data points. Lastly, we performed normalization on our numerical features and generated one hot encodings on our categorical features to facilitate our modeling efforts.
# MAGIC
# MAGIC The final dataset contained 95,250 rows for the 3 month data and 300,772 for the 1 year data. Below is the final list of features broken out by data type:
# MAGIC
# MAGIC
# MAGIC **Numerical**
# MAGIC | Column Name | Description |
# MAGIC |----|----|
# MAGIC | DISTANCE | Distance between airports (miles). |
# MAGIC | ELEVATION | The height of the highest point of an aircraft's landing area |
# MAGIC | HourlyDryBulbTemperature | Hourly dry-bulb temperature, which is commonly used as the standard air temperature reported. |
# MAGIC | HourlyPrecipitation | Amount of precipitation in inches to hundredths over the past hour. A 'T' indicates trace amounts of precipitation. |
# MAGIC | HourlyRelativeHumidity | The hourly relative humidity given to the nearest whole percentage. |
# MAGIC | HourlyStationPressure | Hourly atmospheric pressure observed at the station during the time of observation. Given in inches of Mercury (in Hg). |
# MAGIC | HourlyVisibility | Hourly visibility. This is measured as the horizontal distance an object can be seen and identified given in whole miles. Note visibilities less than 3 miles are usually given in smaller increments (e.g. 2.5). |
# MAGIC | HourlyWindSpeed | Speed of the wind at the time of observation (hourly) given in miles per hour (mph). |
# MAGIC | HoursSinceLastFlight | Elapsed time between aircraft's previous arrival time and aircraft's next departure time |
# MAGIC | PrevFlightDelay | Departure Delayed Time of an aircraft's previous flight |
# MAGIC | dep_capped_delay **(Y variable)**| Number of minutes between scheduled time and actual time. Null if flight was canceled. 0 if on time or early. Capped by it's 95th percentile |
# MAGIC
# MAGIC **Categorical**
# MAGIC | Column Name | Description |
# MAGIC |----|----|
# MAGIC | ORIGIN | Origin Airport |
# MAGIC | DEST | Destination Airport |
# MAGIC | OP_CARRIER | Airline Carrier |
# MAGIC | departure_time_buckets |  Departure time by Morning, Afternoon, Evening, or Night |
# MAGIC | DAY_OF_WEEK | Day of Week |
# MAGIC
# MAGIC **Boolean**
# MAGIC | Column Name | Description |
# MAGIC |----|----|
# MAGIC | is_holiday | Flag if flight is departing on a holiday |
# MAGIC | is_extreme_weather | Flag if flight departing under extreme weather |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC In Phase 1, we did an initial exploratory analysis to familiarize ourselves with the data we planned to utilize. All of the EDA is performed on the 1 year (Jan-Dec 2015) OTPW dataset which contains 5,811,854 flights. Below is the summary statistics of our target regression variable: `DEP_DELAY_NEW`.
# MAGIC
# MAGIC | Column | Mean | Std | Min | 25% | 50% | 75% | Max |
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | DEP_DELAY_NEW | 32.6 | 52.4 | 1.0 | 5.0 | 15.0 | 38.0 | 1988.0 |
# MAGIC
# MAGIC
# MAGIC Of the 216 columns, a significant portion had too many null fields to be of use, so only fields with less than 10% of null values where considered for analysis. Of those that remained, we have selected a handful of columns which logically seem to be valid contributors to flight delays.
# MAGIC
# MAGIC ![day of month](https://github.com/riverliway/261-final/blob/master/day_of_month.png?raw=true)
# MAGIC
# MAGIC There does not appear to be any particular day or group of days during each month that has more or less delays, the average delay appears fairly uniform.
# MAGIC
# MAGIC ![day of week](https://github.com/riverliway/261-final/blob/master/day_of_week.png?raw=true)
# MAGIC
# MAGIC Monday has slightly higher delays than the rest of the week, but not by a large margin.
# MAGIC
# MAGIC ![month](https://github.com/riverliway/261-final/blob/master/month.png?raw=true)
# MAGIC
# MAGIC June and December boast the highest delay months, likely from school breaks when more families will travel for the holidays.
# MAGIC
# MAGIC ![dep time](https://github.com/riverliway/261-final/blob/master/dep_time.png?raw=true)
# MAGIC
# MAGIC The scheduled departure time shows us that there are less delays early in the morning while there are much longer delays in the evening.
# MAGIC
# MAGIC ![arr time](https://github.com/riverliway/261-final/blob/master/arrival_time.png?raw=true)
# MAGIC
# MAGIC The scheduled arrival time shows a similar story, although slightly offset. We hypothesize that delays are more commonly associated with higher number of flights, so early mornings are when the least number of flights are taking off at the same time while evenings has the most number of flights taking off.
# MAGIC
# MAGIC ![airlines](https://github.com/riverliway/261-final/blob/master/carrier.png?raw=true)
# MAGIC
# MAGIC The top airlines with the most delays are Frontier and Spirit.
# MAGIC
# MAGIC ![origin](https://github.com/riverliway/261-final/blob/master/departing.png?raw=true)
# MAGIC
# MAGIC The origin airport with the highest departing delays is Wilmington Airport in New Castle, Delaware.
# MAGIC
# MAGIC ![dest](https://github.com/riverliway/261-final/blob/master/arriving.png?raw=true)
# MAGIC
# MAGIC Similarly, the destination airport with the highest delays is also  Wilmington Airport in New Castle, Delaware.
# MAGIC
# MAGIC ![location](https://github.com/riverliway/261-final/blob/master/location.png?raw=true)
# MAGIC
# MAGIC By plotting the delays on a map, we can see that location doesn't seem to have a strong relationship with delays. There isn't a particular trend with cardinal directions or administrative districts. We do have the data for the US's pacific islands included in our analysis, but have excluded it from this map for illistrative purposes.
# MAGIC
# MAGIC ![distance](https://github.com/riverliway/261-final/blob/master/distance.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and distance.
# MAGIC
# MAGIC ![temperature](https://github.com/riverliway/261-final/blob/master/temperature.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and temperature.
# MAGIC
# MAGIC ![altitude](https://github.com/riverliway/261-final/blob/master/altitude.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and altitude.
# MAGIC
# MAGIC ![percipitation](https://github.com/riverliway/261-final/blob/master/precipitation.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and percipitation.
# MAGIC
# MAGIC ![humidity](https://github.com/riverliway/261-final/blob/master/humidity.png?raw=true)
# MAGIC
# MAGIC There is a parabola relationship apparent between delays and humidity.
# MAGIC
# MAGIC ![pressure](https://github.com/riverliway/261-final/blob/master/pressure.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and pressure.
# MAGIC
# MAGIC ![visibility](https://github.com/riverliway/261-final/blob/master/visibility.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and visibility.
# MAGIC
# MAGIC ![wind](https://github.com/riverliway/261-final/blob/master/wind.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and wind speed.
# MAGIC
# MAGIC Since many of the weather events have an abundance of noise and only trigger delays in extreme cases, we have decided to turn several of these continious variables into discrete ones by bucketing them in severity.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Algorithms and Metrics
# MAGIC
# MAGIC Taking into account EDA to inform intentional feature selection and feature engineering, we implemented our first ML pipeline on the data. The goal of this Phase's analysis is to us a simple ML algorithm, linear regression, to develop a solid pipeline, strongly engineered features, and robust time-series cross validation. In Phase 3, we will build on this progress by testing models more complex than Linear Regression. 
# MAGIC
# MAGIC **Primary Model: Ordinary Least Squares (OLS)**
# MAGIC    - **Implementation:** PySpark (`pyspark.ml.regression.LinearRegression`)
# MAGIC    - Explanation: OLS linear regression is the most common version of linear regression. It estimates the relationship between independent variables and the dependent variable by minimizing the sum of the squared differences between the observed and predicted values.
# MAGIC
# MAGIC **Loss Function:** Root Mean Squared Log Error (RMSLE)
# MAGIC
# MAGIC **Note on Regularization:** Regularization was used in initial experiments but dropped from this stage of the analysis due to no observed benefit. Regularization can be used to help prevent overfitting, a problem which we have not observed. We will revisit regularization in subsequent modeling as needed.
# MAGIC
# MAGIC #### Description of Metrics
# MAGIC - **Root Mean Squared Error (RMSE)**
# MAGIC    - RMSE is a common performance metric for linear regression and was used at this early stage for ease of testing. In the future, our team will implement RMSLE, as well (rationale below).
# MAGIC    - RMSE Formula:
# MAGIC      \\(
# MAGIC      \text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
# MAGIC      \\)
# MAGIC
# MAGIC - Secondery metric:  **Mean Average Error (MAE)**
# MAGIC    - Whereas RMSE increases the penalty for large errors, MAE simply takes the average of all errors.
# MAGIC    - Formula:
# MAGIC      \\(
# MAGIC      \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left|y_i - \hat{y}_i\right|
# MAGIC      \\)
# MAGIC
# MAGIC ##### Additional Metrics to be Explored in Phase 3: 
# MAGIC - **Root Mean Squared Log Error (RMSLE)**
# MAGIC    - Used for evaluating regression models. Using RMSLE is useful in that it penalizes underestimations far more than overestimations. In the context of predicting flight delays: for a true delay time of 1 hour, we would penalize the model more for predicting a 15 min delay than we would for predicting a 2 hour delay.
# MAGIC    - Formula:
# MAGIC      \\(
# MAGIC      \text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}
# MAGIC      \\)
# MAGIC
# MAGIC - **Mean Average Percentage Error (MAPE)**
# MAGIC    - Common in forecasting, the MAPE metrics looks at the percentage of error from the forecast to actual value. This metric would, in effect, treat a 15 minute error on a 150 minute delay the same as a 2 minute error on a 20 minute delay. 
# MAGIC    - Formula:
# MAGIC      \\(
# MAGIC       \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
# MAGIC      \\)
# MAGIC
# MAGIC **Note:** For this regression task, a prediction of -10 mins delay was treated as a prediction of on-time (0 mins delay).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Machine Learning Pipeline
# MAGIC
# MAGIC We implemented a number of machine learning pipelines, the details of which can be reviewed in the experiments table. The below graphic depicts all steps of the pipeline. In some cases, cross validation was carried out. In others, plain linear regression was used. We experimented with downsampling (the 4th step) but did not use this technique for every model. A large portion of this phase was spent iterating between feature engineering and model training / evaluation.
# MAGIC
# MAGIC ##### ML Pipeline
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/ML_pipeline1.jpg?raw=True)
# MAGIC
# MAGIC ### Pipeline Stages Description
# MAGIC
# MAGIC 1. **EDA & Data Ingestion**
# MAGIC    - **Description:** This stage involves loading flight and weather data, and conducting Exploratory Data Analysis (EDA) to understand data structure, identify patterns, detect anomalies, and summarize characteristics. Key activities include importing datasets from Azure Blob Storage, understanding data distribution and relationships, identifying missing values and outliers, and visualizing data (e.g., delays over time, weather impact). See above EDA section for details.
# MAGIC
# MAGIC 2. **Data Cleaning**
# MAGIC    - **Description:** Preparing the data for analysis by addressing issues identified during EDA to ensure accuracy and consistency. Key activities include handling missing data, removing or correcting outliers, standardizing and normalizing data, and encoding categorical variables (e.g., airport codes, airlines).
# MAGIC
# MAGIC 3. **Feature Engineering**
# MAGIC    - **Description:** Creating new features from existing data by transforming raw data into meaningful inputs. Key activities include creating new features, transforming existing features (e.g., scaling delays), generating interaction features, and reducing dimensionality (i.e. PCA) if necessary. See above feature engineering section for details.
# MAGIC
# MAGIC 4. **Down Sampling**
# MAGIC    - **Description:** In this case, downsampling was necessary given the large number of observations with delay time of 0. When downsampling, we randomly eliminated 80% of observations with delay time of 0 minutes. 
# MAGIC
# MAGIC 5. **Time Series Splitting and Linear Regression**
# MAGIC    - **Description:** The data was split by date into train and validation sets. Both datasets were split by date in various ways depending on the experiment, always in a train/val split between 65/35 and 80/20. OLS Linear Regression was used to train a model on the train data. 
# MAGIC
# MAGIC 6. **TimeSeries CV Splitting and Linear Regression with Cross Validation**
# MAGIC    - **Description:** Rolling and blocked time-series cross validation were set up on top of our linear model. The flow of data through the cross validation process is shown in the table below. More information on rolling and blocked cross validation can be found [here.](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20on%20Time%20Series,for%20the%20forecasted%20data%20points)
# MAGIC
# MAGIC ##### Cross Validation Flow 
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/cv_pipeline1.jpg?raw=True)
# MAGIC
# MAGIC 7. **Model Evaluation**
# MAGIC    - **Description:** Assessing trained models on the validation dataset to ensure they generalize well to unseen data. Key activities include evaluating models using metrics (e.g. RMSE, MAE), comparing performance, analyzing errors, visualizing performance, and iterating on feature engineering, downsampling, model selection, etc.
# MAGIC
# MAGIC 8. **Model Selection**
# MAGIC    - **Description:** Selecting the best-performing model based on evaluation metrics and preparing it for deployment
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Training Data 
# MAGIC
# MAGIC Cleaned data with all engineered features was pre-stored in blob storage for ease of modeling. The cleaned 3 month dataset (Jan-Mar 2015) contained 95,250 observations and the cleaned 1 year dataset (2015) contained 300,772 observations. 
# MAGIC
# MAGIC Despite a relatively short list of features, the number of actual features used in the model is quite large (600+) due to the use of one-hot encodings, specifically of the DEST / ORIGIN features.
# MAGIC
# MAGIC ##### Number of Total Input Features 
# MAGIC | Dataset | Count |
# MAGIC | ----- | ----- |
# MAGIC | 3 mo | 629 |
# MAGIC | 1 yr | 653 | 
# MAGIC
# MAGIC Chosen and engineered features and the datatypes of each are specified below:
# MAGIC
# MAGIC ##### Feature Breakdown by Family and Count
# MAGIC | Feature Family | Encoding | Count of Features in 3 mo Dataset | Count of Features in 1 yr Dataset |
# MAGIC | ---------- | ---------- | ---------- | ---------- |
# MAGIC | DAY_OF_WEEK | one-hot | 6 | 6 |
# MAGIC | Departure_Time_Buckets | one-hot | 3 | 3 |
# MAGIC | OP_CARRIER | one-hot | 13 | 13 |
# MAGIC | DEST | one-hot | 311 | 321 |
# MAGIC | ORIGIN | one-hot | 283 | 297 |
# MAGIC | is_extreme_weather | boolean | 1 | 1 |
# MAGIC | is_holiday | boolean | 1 | 1 |
# MAGIC | is_vert_obscured | boolean | 1 | 1 |
# MAGIC | DISTANCE | scaled numeric | 1 | 1 |
# MAGIC | ELEVATION | scaled numeric | 1 | 1 |
# MAGIC | Weather Parameters (HourlyDryBulbTemperature, HourlyPrecipitation, HourlyRelativeHumidity, HourlyStationPressure, HourlyVisibility, HourlyWindSpeed) | scaled numeric | 6 | 6 |
# MAGIC | HoursSinceLastFlight | scaled numeric | 1 | 1 |
# MAGIC | PrevFlightDelay | scaled numeric | 1 | 1 |
# MAGIC
# MAGIC
# MAGIC ### Modeling Environment
# MAGIC All modeling was completed in a DataBricks cluster with a standard_DS4_v2 28GB driver with 8 cores, and 2-6 standard_DS3_v2 workers with 28-84GB and 8-24 cores. In this environment, each model took less than 1 minute to train, even on the 1 yr dataset.
# MAGIC
# MAGIC ### Experiments
# MAGIC
# MAGIC In total, 12+ experiments were conducted. Highlighted experiments can be seen in the below table. Details were summarized on a google sheet [here.](https://docs.google.com/spreadsheets/d/1fcTjJKpwbI-GD0_ITRqAArBrXO3yETHPUMJImPrzJkk/edit?usp=sharing) Additionally, models can be accessed and reviewed through MLFlow.
# MAGIC
# MAGIC
# MAGIC ##### Experiment Summary
# MAGIC
# MAGIC Model No. | Dataset | Model Summary | Model Type | Features (X)| Label (Y) | Split | Train RMSE | Validation RMSE |
# MAGIC | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
# MAGIC | 0 | 3mo | BASELINE | Average | NA | "DEP_DELAY_NEW" | first 2 months train, 3rd month test | 51.0 | 51.0 |
# MAGIC | 4 | 3mo | Linear model with all engineered features added. | Linear Regression | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER', 'DEST', 'ORIGIN', 'DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "DEP_DELAY_NEW" | first 2 months train, 3rd month test | 44.688 | 48.01 |
# MAGIC | 5 | 3mo | Linear model with DEST and ORIGIN removed. | Linear Regression | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "DEP_DELAY_NEW" | first 2 months train, 3rd month test | 45.21 | 47.91 |
# MAGIC | 7 | 3mo | Linear model with delay departure minutes capped to reduce long tail. | Linear Regression | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "dep_capped_delay" | train_cutoff_date = "2015-03-15" | 18.95 | 18.1 |
# MAGIC | 8 | 3mo | Cross-validated linear model with same configuration as model 7. | Linear Regression w. Time-Series CV | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "dep_capped_delay" | rolling splits with train_cutoff_date = "2015-03-15" | 18.44 | 18.21 |
# MAGIC | 9 | 3mo |  Same configuration as model 7 with downsampled dataset. | Linear Regression | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "dep_capped_delay" | train_cutoff_date = "2015-03-15" | 21.47 | 21.87 |
# MAGIC | 11 | 1yr |  Same configuration as model 7 on larger dataest. | Linear Regression | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "dep_capped_delay" | train_cutoff_date = "2015-10-31" | 18.79 | 18.49 |
# MAGIC | 12 | 1yr |  Cross-validated linear model with same configuration as model 11. | Linear Regression w. Time-Series CV | ['DAY_OF_WEEK','Departure_Time_Buckets', 'OP_CARRIER','DISTANCE', 'ELEVATION', Weather Parameters, 'HoursSinceLastFlight', 'PrevFlightDelay','is_extreme_weather', 'is_holiday', 'is_vert_obscured'] | "dep_capped_delay" | rolling splits with train_cutoff_date =  "2015-10-31" | 18.74 | 18.54 |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and Discussion
# MAGIC
# MAGIC The best performing models from the experimentats table above are models 7 and 8 from the 3 month dataset and models 11 and 12 from the 1 year dataset. All of these models have the same configuration, with the only difference being the dataset size and whether cross-validation was used. Compared to our baseline RMSE of 51.0, train/validation RMSE of 18.XX are a massive improvement. Still, there is more work to be done to improve our models.
# MAGIC
# MAGIC **Observations on Feature Importance:** For each experiment conducted, we calculated the absolute value of the coefficients as a proxy for relative feature importance. Surprisingly, even after adding numerous thoughtfully engineered features, the top-ranking features were always from DEST or ORIGIN feature families. That is, the model is using the destination or origin airports to a large extent in it's prediction of a flight's delay time. This leads us to believe that some airports just can't get their act together! In a hyopthetical 'Phase 4' of this project, it could make sense to include features that can better indicate whether a specific airport will be experiencing delays that day (i.e. logistics data such as number of planes on tarmac, staffing information, etc. ).
# MAGIC
# MAGIC To better gauge the relevance of other features, we tested models without the DEST and ORIGIN features included. The features with the highest and lowest magnitude coefficients are represented in the charts below. We can see that departure times in the afternoon and evening have high relative importance, which tracks with our EDA results. Previous flight delay is the third most high-ranking parameter, reaffirming our thought process when creating this feature. 
# MAGIC
# MAGIC Other features we created, however, did not rank as highly as we expected. 'is_holiday' is among the lowest ranking features, but we know from experience how much holidays can impact airport travel plans. One possible reason for this is because, due to the nature of the 1 yr dataset, we were unable to include observations for Christmas or Thanksgiving in our training set. Upon training the larger dataset, we may see is_holiday increase its coefficient. 
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/feat_import.png?raw=True)
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/feat_import.png?raw=True)
# MAGIC
# MAGIC
# MAGIC **Observations on Delay Departure Data Distrubution:** The target parameter, departure delays, had a left skewed distribution with a long tail. Many observations were equal to 0, yet some reached all the way past 2000 minutes. To mitigate the problem of non-normal distribution, we opted to cap the delay minutes, so all delays that exceeded a certain threshold were lumped into a single bucket. This change, alone, had a massive impact on performance, bringing our RMSE scores from 45-47 min range down to the 18-19 min range.
# MAGIC
# MAGIC **Observations on Downsampling:** When downsampling, we randomly eliminated 80% of observations with delay time of 0 minutes. This massive cut still left '0' as the majority delay time, and the downsampling attempt actually hurt model performance. In future models we may experiment with eliminating more or all values with departure delay = 0 to get a better idea of the trend here. 
# MAGIC
# MAGIC **Time-Series Cross Validation:** The time-series cross validation models did not perform markedly better than standard linear regression models. This makes sense given the limitations for time series cross validation. We expect cross-validation to be more useful for hyperparameter tuning with more complex models. 
# MAGIC
# MAGIC When comparing blocked to rolling cross validation, rolling splits performed markedly better than blocked. This may be because of the larger datasets that are used to train in rolling splits (and in this scenario, our dataset is limited to begin with). However, when using rolling splits we must be wary of data leakage.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC This project focused on predicting domestic flight delays for US airlines using machine learning, a critical task for improving passenger satisfaction and operational efficiency. Our hypothesis was that regression models incorporating custom features can serve as a good first step in estimating flight delays.
# MAGIC
# MAGIC We utilized historical flight data from the U.S. DOT and weather data from NOAA. We created features based on flight schedules, weather conditions, previous flight delays, and holidays, converting categorical data into one-hot encodings and normalizing numerical features. We developed and evaluated linear regression models with cross-validation, using RMSE and MAE metrics for evaluation. 
# MAGIC
# MAGIC Our models showed improvements over baseline predictions. For instance, our baseline model had an RMSE of 51 minutes. By adding engineered features, we reduced the test RMSE to 48 minutes. Further refinement, such as capping delay minutes and implementing cross-validation, brought the test RMSE down to 18 minutes
# MAGIC
# MAGIC Future work includes revisiting EDA and feature engineering, implementing RMSLE and MAPE metrics, and exploring nonlinear models like XGBoost and MLP Neural Networks.

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
# MAGIC #### Phase 2
# MAGIC | Phase | Engineer | Description | Time Estimation |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | 1 | River | EDA | 6 hours |
# MAGIC | 1 | Nick + Abdul | Feature Engineering | 10 hour |
# MAGIC | 1 | Darby + Abdul | ML Pipelines & Experimentation | 15 hour |
# MAGIC | 1 | Everyone | Assembling Presentation | 3 hour |
# MAGIC | 1 | Everyone | Assembling Report | 8 hour |
# MAGIC
# MAGIC #### Expected Phase 3
# MAGIC | Phase | Engineer | Description | Time Estimation |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | 3 | Everyone | XGB and MLP Modeling | 20 hour |
# MAGIC | 3 | Everyone | Final ML Pipeline| 10 hour |
# MAGIC | 3 | Everyone | Assembling the deliverable | 5 hour |

# COMMAND ----------

