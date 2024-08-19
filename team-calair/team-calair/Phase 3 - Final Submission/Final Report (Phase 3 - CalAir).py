# Databricks notebook source
# MAGIC %md
# MAGIC ##CalAir: Predicting Domestic Flight Delays Using Machine Learning
# MAGIC Time-series Machine Learning Project by the CalAir Data Dcience Team (Team 2-2)
# MAGIC
# MAGIC ## Phase 3 Leader Plan
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
# MAGIC Airline delays represents one of the largest cost factors for airports and are one of the most common painpoints for flyers around the world. CalAir is has developed regression models to predict flight delay duration in minutes for US domestic flights. Our models are based on a prepared dataset that incorporates historical flight and weather data from 2015 to 2019. Flight data is sourced from the US flights Department of Transportation and and weather data from the National Oceanic and Atmospheric Administration. 
# MAGIC
# MAGIC Over the course of 5 weeks, CalAir conduted in-depth EDA, feature engineering, and modeling with linear regression, XGBoost, and Multi-Layer Perceptron models. We chose standard Root Mean Squared Error (RMSE) as loss function, and as a metric to assess the impact of our modeling choices on performance. RMSE penalizes large errors more than smaller ones, reaulting in models that are less likely to be 'way off' in their predicted delay time. RMSE is also interpretable in the same units as our metric (minutes), providing a reasonable representation of how severe the error in delay minutes predicted would be, on average. In addition to RMSE, we reframe this task as categorization to evaluate F3 scores as secondary decision criteria. The F3 metric for categorization balances recall and precision, while prioritizing recall. Using F3, we are able to select a model with fewer false negatives (predicting a delay of <15 minutes when the true delay is >15 minutes). We prefer this type of performance in order to set expectations with customers if their flight is at risk of delay. 
# MAGIC
# MAGIC After comparing RMSE and F3 metrics across 30+ experiments, we have identified the XGBoost model with engineered features added as the best-performing model, with an RMSE of 15. This is a large improvement over our baseline RMSE of 62 minutes, which was calculated by predicting the average of all delay times. 
# MAGIC  
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Introduction 
# MAGIC
# MAGIC Over the past decade, flight delays and cancellations have become [increasingly problematic for travelers](https://www.cbsnews.com/news/the-future-of-flying-more-delays-more-cancellations-more-chaos/). In this shadow of decreased operational efficiency, the ability to accurately forecast the length of a delay is critical. This not only improves customer satisfaction by providing more reliable travel information, but also assists airlines and airports in optimizing their operations.
# MAGIC
# MAGIC When operating or boarding a flight, two pieces of information are critical: (1) will there be a delay? and (2) how long will that delay be? CalAir has developed and evaluated various machine learning models to answer question 2--predict the length of flight delays for airlines in terms of minutes of delay from scheduled departure time. Given this objective, we focus only on flights that departed successfully while dropping flights that were cancelled or diverted. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data
# MAGIC
# MAGIC #### Data Description
# MAGIC
# MAGIC We used the following datasets to develop the models. These data have been pre-joined for direct comparison in a dataset we call OTPW (ontime performance of flights and weather).
# MAGIC
# MAGIC - **Flights Data:**
# MAGIC   - Source: [U.S. Department of Transportation (DOT) Bureau of Transportation Statistics](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)
# MAGIC   - Dimensions: 31,746,841 x 109 (for 2015-2019)
# MAGIC   - Subsets: Q1 2015, Q1+Q2 2015, 2019, 2015-2019
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`
# MAGIC   - Summary: Sourced from the Bureau of Transportation Statistics, this dataset includes a quantitative summary of flights' departure performance, arrival performance, cancellations and diversions. Additional features represent the carrier, dates/times, origin/destination information, and reason for delay. The data has been parsed into small subsets which were used for development. Initial model pipelines were developed with 3-month and 12-month dataset from 2015 and, susequently, on a 5-year dataset from 2015-2019. A best-performing model pipeline will be trained and tested on the complete dataset (2015-2021).
# MAGIC
# MAGIC - **Weather Data:**
# MAGIC   - Source: [National Oceanic and Atmospheric Administration (NOAA)](https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf)
# MAGIC   - Dimensions: 630,904,436 x 177 (for 2015-2019)
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/`
# MAGIC   - Summary: National weather data is included to supplement the flights delay data for 2015-2019, as a significant proportion of delays were attributed to weather. This dataset includes daily and monthly summary statistics as well as hourly data for precipitation, visibility, wind speed, etc. 
# MAGIC
# MAGIC - **Airport Metadata:**
# MAGIC   - Source: US Department of Transportation (DoT)
# MAGIC   - Dimensions: 18,097 x 10
# MAGIC   - Location: `dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data`
# MAGIC   - Summary: This dataset provides relevant metadata about each airport.
# MAGIC
# MAGIC - **Airport Codes:**
# MAGIC   - Source: [DataHub](https://datahub.io/core/airport-codes)
# MAGIC   - Summary: This table contains IATA or ICAO airport codes that was joined to the flights dataset to represent each airport properly.
# MAGIC
# MAGIC
# MAGIC #### Data Cleaning and Preprocessing
# MAGIC
# MAGIC The data used in this project was pre-joined, so our focus was cleaning the provided OTPW dataset. Key steps of the pipeline for data cleaning, preprocessing, and feature engineering can be seen below: 
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/data_preprocessing.jpg?raw=True)
# MAGIC
# MAGIC For data cleaning, we assigned the proper data types to each column to help with future transformations down the road. We also filtered out flights that were cancelled and diverted and dropped rows where we had empty data because we couldn't infer anything from the missing data points. For this regression task, a prediction of -10 mins delay was treated as a prediction of on-time (0 mins delay) and the data was adjusted accordingly.
# MAGIC
# MAGIC For data transformation and feature engineering, we selected features initially based on those that had less than 10% missing data and, intuitively, were possibly related to airport delays.  We conducted feature engineering on pre-split data in order to reduce leakage. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC After data cleaning and preprocessing, we conducted exploratory data analysis to familiarize ourselves with the data we planned to utilize. Primary EDA was performed on the 4 years (2015-2018) OTPW dataset which contains 24,279,321 flights. The final year of the dataset (2019) was held out for EDA as it will act as our test set. Any flights that were delayed or canceled have been removed from the EDA and training & test sets. Below are summary statistics of our target regression variable: `DEP_DELAY_NEW`.
# MAGIC
# MAGIC | Column | Mean | Std | Min | 25% | 50% | 75% | Max |
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | DEP_DELAY_NEW | 18.3 | 52.1 | 0.0 | 0.0 | 0.0 | 13.0 | 1988.0 |
# MAGIC
# MAGIC
# MAGIC Of the 214 columns, a significant portion had too many null fields to be of use, so only fields with less than 10% null values where considered for analysis. Of those that remained, we have selected a handful of columns which logically seem to be valid contributors to flight delays.
# MAGIC
# MAGIC ![date](https://github.com/riverliway/261-final/blob/master/date2.png?raw=true)
# MAGIC
# MAGIC We can definitely see seasonality trends where the delays spike in summers and springs, most likely because that is when families go on vacation. 
# MAGIC
# MAGIC ![day of month](https://github.com/riverliway/261-final/blob/master/day_of_month2.png?raw=true)
# MAGIC
# MAGIC There does not appear to be any particular day or group of days during each month that has more or less delays, the average delay appears fairly uniform.
# MAGIC
# MAGIC ![day of week](https://github.com/riverliway/261-final/blob/master/day_of_week2.png?raw=true)
# MAGIC
# MAGIC Monday, Thursday, and Friday have slightly higher delays than the rest of the week, but not by a large margin.
# MAGIC
# MAGIC ![month](https://github.com/riverliway/261-final/blob/master/month2.png?raw=true)
# MAGIC
# MAGIC The summer months of June, July, and August boast the highest delay months, likely from school breaks when more families will travel for the holidays.
# MAGIC
# MAGIC ![year](https://github.com/riverliway/261-final/blob/master/year2.png?raw=true)
# MAGIC
# MAGIC We see a slight increase in delays over time.
# MAGIC
# MAGIC ![dep time](https://github.com/riverliway/261-final/blob/master/dep_time2.png?raw=true)
# MAGIC
# MAGIC The scheduled departure time shows us that there are less delays early in the morning while there are much longer delays in the evening.
# MAGIC
# MAGIC ![arr time](https://github.com/riverliway/261-final/blob/master/arrival_time2.png?raw=true)
# MAGIC
# MAGIC The scheduled arrival time shows a similar story, although slightly offset. We hypothesize that delays are more commonly associated with higher number of flights, so early mornings are when the least number of flights are taking off at the same time while evenings has the most number of flights taking off.
# MAGIC
# MAGIC ![airlines](https://github.com/riverliway/261-final/blob/master/carrier2.png?raw=true)
# MAGIC
# MAGIC The top airlines with the most delays are Frontier, JetBlue, and Allegiant.
# MAGIC
# MAGIC ![origin](https://github.com/riverliway/261-final/blob/master/departing2.png?raw=true)
# MAGIC
# MAGIC The origin airport with the highest departing delays is Youngstown-Warren Regional Airport in Ohio.
# MAGIC
# MAGIC ![dest](https://github.com/riverliway/261-final/blob/master/arriving2.png?raw=true)
# MAGIC
# MAGIC Similarly, the destination airport with the highest delays is also  Youngstown-Warren Regional Airport.
# MAGIC
# MAGIC ![location](https://github.com/riverliway/261-final/blob/master/location.png?raw=true)
# MAGIC
# MAGIC By plotting the delays on a map, we can see that location doesn't seem to have a strong relationship with delays. There isn't a particular trend with cardinal directions or administrative districts. We do have the data for the US's pacific islands included in our analysis, but have excluded it from this map for illistrative purposes.
# MAGIC
# MAGIC ![distance](https://github.com/riverliway/261-final/blob/master/distance2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and distance.
# MAGIC
# MAGIC ![temperature](https://github.com/riverliway/261-final/blob/master/temperature2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and temperature.
# MAGIC
# MAGIC ![altitude](https://github.com/riverliway/261-final/blob/master/alitude2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and altitude.
# MAGIC
# MAGIC ![percipitation](https://github.com/riverliway/261-final/blob/master/precipitation2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and percipitation.
# MAGIC
# MAGIC ![humidity](https://github.com/riverliway/261-final/blob/master/humidity2.png?raw=true)
# MAGIC
# MAGIC There is a parabola relationship apparent between delays and humidity.
# MAGIC
# MAGIC ![pressure](https://github.com/riverliway/261-final/blob/master/pressure2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and pressure.
# MAGIC
# MAGIC ![visibility](https://github.com/riverliway/261-final/blob/master/visibility2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and visibility.
# MAGIC
# MAGIC ![wind](https://github.com/riverliway/261-final/blob/master/wind_speed2.png?raw=true)
# MAGIC
# MAGIC There isn't a strong relationship apparent between delays and wind speed.
# MAGIC
# MAGIC Since many of the fields we explored in the exploratory data analysis don't appear to have a strong relationship by themselves, or the trend is obscured by noise, we have performed extra data cleaning and created synthetic features to improve our model's performance.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Engineering
# MAGIC
# MAGIC A large proportion of our model performance was achieved through thoughtful feature engineering. We developed numerous features that could help explain the behavior behind flight delays. Below is the list of features and our logic for designing them:
# MAGIC | New Feature Name | Logic |
# MAGIC |----|----|
# MAGIC | is_vert_obscured | Based on the hourly sky condition free text, we parsed for the code "VV" which indicated vertical obscurity. This was added based on domain expertise as to what weather conditions cause delay. |
# MAGIC | departure_time_buckets | Categorized CRS_DEP_TIME by Morning, Afternoon, Evening, or Night. Typically, we observed flights later in the day had more delays. |
# MAGIC | HoursSinceLastFlight | Based on tail number, we looked at the aircraft's previous arrival time and took the total elapsed time between that and it's next departure time. The turnaround time between flights has implications for 'snowballing' delays. |
# MAGIC | PrevFlightDelay | Based on tail number, we grab the aircraft's previous flight departure delay. An aircraft arriving late from a previously delayed flight has strong implications for subsequent delays. |
# MAGIC | is_extreme_weather | Using the hourly present weather type free text, we parsed the text for conditions we deemed as extreme weather. This binary variable was created based on the noise of weather data observed in EDA. |
# MAGIC | dep_capped_delay | Given the long tail for departure delay, we generated a new field that would cap the delay by it's 95th percentile. This became our new Y parameter! |
# MAGIC | is_holiday | Flagged flights if the flight date was on a holiday. |
# MAGIC | origin_pagerank | A graph algorithm to represent the flight traffic at the origin airport. Higher-traffic airports may be hubs that have more (or less) operational efficiency. |
# MAGIC | dest_pagerank | A graph algorithm to represent the flight traffic at the destination airport. Higher-traffic airports may be hubs that have more (or less) operational efficiency. |
# MAGIC
# MAGIC #### Page Rank Feature
# MAGIC
# MAGIC Exploring coefficients of our initial linear models highlighted the importance of origin and destination airports in predicting delay time. We hypothesize this is due to airport operations/logistics details that are not available in the current dataset. One step we took to represent airport-level logistics is to implement a page-rank feature. Page rank is a graph based alorithm that ranks the importance of each node (airport) based on its links to other nodes. Here, the links (edges) are flights to/from other airports. A visualization of the graph underlying this algorithm is below:
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/page_rank.jpg?raw=True)
# MAGIC
# MAGIC
# MAGIC ####Final Feature Set 
# MAGIC
# MAGIC Once all features were created, we perform normalization on our numerical features and generated one hot encodings on our categorical features to facilitate our modeling efforts.
# MAGIC
# MAGIC Below is a summary of all features used for downstream modeling:
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
# MAGIC | origin_pagerank | A representation of the flight traffic at the origin airport. |
# MAGIC | dest_pagerank | A representation of the flight traffic at the destination airport. |
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
# MAGIC ### Training Data 
# MAGIC
# MAGIC Cleaned data with all engineered features was pre-stored in blob storage after feature engineering stage. The three datasets used are detailed below:
# MAGIC
# MAGIC | Data Range| # Features (after one-hot) | Train Set | Test Set |
# MAGIC |----|----|----|----|
# MAGIC | 3 month | 631 | 83,052 | 12,198 |
# MAGIC | 1 year | 653 | 233,231 | 67,541 |
# MAGIC | 5 year | 758 | 1,206,987 | 365,428 |
# MAGIC
# MAGIC Despite a relatively short list of features, the number of actual features used in the model is quite large (600+) due to the use of one-hot encodings, specifically of the DEST / ORIGIN features.
# MAGIC
# MAGIC Chosen and engineered features and the datatypes of each are specified below:
# MAGIC
# MAGIC ##### Feature Breakdown by Family and Count
# MAGIC | Feature Family | Encoding | Count of Features in 3 mo Dataset | Count of Features in 1 yr Dataset | Count of Features in 5 yr Dataset |
# MAGIC | ---------- | ---------- | ---------- | ---------- | ---------- |
# MAGIC | DAY_OF_WEEK | one-hot | 6 | 6 | 6 |
# MAGIC | Departure_Time_Buckets | one-hot | 3 | 3 | 3 |
# MAGIC | OP_CARRIER | one-hot | 13 | 13 | 18 |
# MAGIC | DEST | one-hot | 311 | 321 | 365 |
# MAGIC | ORIGIN | one-hot | 283 | 297 | 351 |
# MAGIC | is_extreme_weather | boolean | 1 | 1 | 1 |
# MAGIC | is_holiday | boolean | 1 | 1 | 1 |
# MAGIC | is_vert_obscured | boolean | 1 | 1 | 1 |
# MAGIC | DISTANCE | scaled numeric | 1 | 1 | 1 |
# MAGIC | ELEVATION | scaled numeric | 1 | 1 | 1 |
# MAGIC | Weather Parameters (HourlyDryBulbTemperature, HourlyPrecipitation, HourlyRelativeHumidity, HourlyStationPressure, HourlyVisibility, HourlyWindSpeed) | scaled numeric | 6 | 6 | 6 |
# MAGIC | HoursSinceLastFlight | scaled numeric | 1 | 1 | 1 |
# MAGIC | PrevFlightDelay | scaled numeric | 1 | 1 | 1 |
# MAGIC | dest_pagerank | scaled numeric | 1 | 1 | 1 |
# MAGIC | origin_pagerank | scaled numeric | 1 | 1 | 1 |
# MAGIC
# MAGIC ### Exploratory Data Analysis of Engineered Features
# MAGIC
# MAGIC We can take a look at graphs of the synthetic features to better understand their distributions starting with the Departure Time Buckets.
# MAGIC
# MAGIC ![departure_time_bucket2](https://github.com/riverliway/261-final/blob/master/departure_time_bucket2.png?raw=true)
# MAGIC
# MAGIC The buckets do a good job capturing the complex wave pattern seen in the continuous version of the departure time graph from above and removing the excess noise.
# MAGIC
# MAGIC ![prev_flight_elapsed_time2](https://github.com/riverliway/261-final/blob/master/prev_flight_elapsed_time2.png?raw=true)
# MAGIC
# MAGIC The time since the previous flight using the same tail number departed has a more complex relationship to the current flight being delayed than initially predicted, but there is still a pattern.
# MAGIC
# MAGIC ![prev_flight_elapsed_time2](https://github.com/riverliway/261-final/blob/master/prev_flight_delay_scatter2.png?raw=true)
# MAGIC
# MAGIC The delay of the previous flight using the same tail number has a weak relationship with the current delay. The pearson correlation of these two variables is only 0.34. We have randomly sampled 1% of the training flights to make the scatterplot visually consistent. Since this feature is important to our analysis, we have also included an average delay chart below.
# MAGIC
# MAGIC ![prev_flight_elapsed_time2](https://github.com/riverliway/261-final/blob/master/prev_flight_delay_avg2.png?raw=true)
# MAGIC
# MAGIC This chart does not make the relationship appear any more consistent.
# MAGIC
# MAGIC |  | Mean | Std | Min | 25% | 50% | 75% | Max | 
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | Holiday | 17.4 | 46.7 | 0.0 | 0.0 | 0.0 | 15.0 | 1761.0 |
# MAGIC | Non-Holiday | 18.3 | 52.2 | 0.0 | 0.0 | 0.0 | 13.0 | 1988.0 |
# MAGIC
# MAGIC We can also take a look at the departure delay distributions comparing holidays to non-holidays. Only 3.6% of the flights took place on a holiday. Surprisingly, there wasn't as strong of a difference as we had anticipated.
# MAGIC
# MAGIC |  | Mean | Std | Min | 25% | 50% | 75% | Max | 
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | Extreme Weather | 13.3 | 41.6 | 0.0 | 0.0 | 0.0 | 9.0 | 1023.0 |
# MAGIC | Normal Weather | 18.3 | 52.1 | 0.0 | 0.0 | 0.0 | 13.0 | 1988.0 |
# MAGIC
# MAGIC We can also take a look at the departure delay distributions comparing holidays to non-holidays. Only 0.4% of the flights took place during extreme weather. We assume this number to be small because many flights were canceled because of the extreme weather. Surprisingly, extreme weather actually helped the delay issue.
# MAGIC
# MAGIC |  | Mean | Std | Min | 25% | 50% | 75% | Max | 
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC | Vertical Obscured | 24.7 | 70.9 | 0.0 | 0.0 | 0.0 | 17.0 | 1416.0 |
# MAGIC | Vertical Clear | 18.2 | 51.5 | 0.0 | 0.0 | 0.0 | 13.0 | 1988.0 |
# MAGIC
# MAGIC We can also take a look at the departure delay distributions comparing when the fog obscures visibility above and below the plane. Only 2.4% of the flights took place with low vertical visibility. When the vertical was obscured, it seems to significantly delay flights.
# MAGIC
# MAGIC ![origin_pagerank2](https://github.com/riverliway/261-final/blob/master/origin_pagerank2.png?raw=true)
# MAGIC
# MAGIC Taking a look at the scatterplot for the delays by the pagerank of the origin airport, there does not seem to be a relationship. The pearson correlation of these two variables is only 0.02. We have randomly sampled 1% of the training flights to make the scatterplot readable.
# MAGIC
# MAGIC ![origin_pagerank2](https://github.com/riverliway/261-final/blob/master/origin_pagerank_avg2.png?raw=true)
# MAGIC
# MAGIC We can confirm the lack of relationship by looking at the average for each pagerank value.
# MAGIC
# MAGIC ![origin_pagerank2](https://github.com/riverliway/261-final/blob/master/dest_pagerank_scatter2.png?raw=true)
# MAGIC
# MAGIC Taking a look at the scatterplot for the delays by the pagerank of the destination airport, there does not seem to be a relationship. The pearson correlation of these two variables is only -0.009. We have randomly sampled 1% of the training flights to make the scatterplot readable.
# MAGIC
# MAGIC ![origin_pagerank2](https://github.com/riverliway/261-final/blob/master/dest_pagerank_avg2.png?raw=true)
# MAGIC
# MAGIC We can confirm the lack of relationship by looking at the average for each pagerank value.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Algorithms and Metrics
# MAGIC
# MAGIC Three models were tested and are detailed below: Linear Regression, XGBoost, and Multi-Layer Perceptron.
# MAGIC
# MAGIC Across all predictive models, we prioritized the following loss functions and evaluation metrics for consistency:
# MAGIC
# MAGIC **Loss Function:** RMSE
# MAGIC
# MAGIC **Evaluation Function:** RMSE
# MAGIC
# MAGIC - **Primary Metric:** Root Mean Squared Error (RMSE)
# MAGIC    - We chose standard Root Mean Squared Error (RMSE) as loss function, and as a metric to assess the impact of our modeling choices on performance. RMSE penalizes large errors more than smaller ones, reaulting in models that are less likely to be 'way off' in their predicted delay time. RMSE is also interpretable in the same units as our metric (minutes), providing a reasonable representation of how severe the error in delay minutes predicted would be, on average. 
# MAGIC    - RMSE Formula:
# MAGIC      \\(
# MAGIC      \text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
# MAGIC      \\)
# MAGIC
# MAGIC - **Secondery metric:** F3 Score
# MAGIC    - In addition to RMSE, we reframe the results of our regression task as categorization to evaluate F3 scores as secondary decision criteria. The F3 metric for categorization balances recall and precision, while prioritizing recall. Using F3, we are able to select a model with fewer false negatives (predicting a delay of <15 minutes when the true delay is >15 minutes). We prefer this type of performance in order to set expectations with customers if their flight is at risk of delay.
# MAGIC    - Formula:
# MAGIC       \\(
# MAGIC       \text{F3} = \frac{TP}{TP + 0.1 \cdot FP + 0.9 \cdot FN}
# MAGIC       \\)
# MAGIC
# MAGIC **Note on Regularization:** Regularization was used in initial experiments but dropped from this stage of the analysis due to no observed benefit. Regularization can be used to help prevent overfitting, a problem which we have not observed. We will revisit regularization in subsequent modeling as needed.
# MAGIC
# MAGIC #### Baseline: 
# MAGIC
# MAGIC In addition to reviewing RMSE and F3, we compare the RMSE and F3 scores of our model to a **baseline**. Our team took the average departure delay in minutes as the baseline model. Baseline for the 3 month, 1 year, and 5 year sets is below: 
# MAGIC
# MAGIC | Dataset | Baseline Train RMSE | Baseline Test RMSE | Baseline Train F3 | Baseline Test F3 |
# MAGIC | ----- | ----- | ----- | ----- | ----- |
# MAGIC | 3 month | 50.791 | 50.791 | 0.192 | 0.192 |
# MAGIC | 1 year | 51.450 | 51.449 | 0.184 | 0.184 |
# MAGIC | 5 year | 52.051 | 61.667 | 0.173 | 0.180 |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Machine Learning Pipeline
# MAGIC
# MAGIC We implemented a number of machine learning pipelines to improve upon the baseline model. The details of these can be reviewed in the experiments table and the "Model Experiments" section. The below graphic depicts all steps of the pipeline. In many cases, cross validation was carried out as a way to measure model stability. To train the actual deployable model, a simple time-based train/test split was used. A large portion pipeline optimization effort was spent iterating between feature engineering and model training / evaluation.
# MAGIC
# MAGIC ##### ML Pipeline
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/pipeline_phase3final.jpg?raw=True)
# MAGIC
# MAGIC
# MAGIC ##### Cross Validation for Model Stability 
# MAGIC
# MAGIC Blocked time-series cross validation was used for each model type to evaluate the reliability of that model over time (whether time-based predictions can stand up to issues such as data drift). The flow of data through the cross validation process is shown in the table below. More information on blocked cross validation can be found [here.](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20on%20Time%20Series,for%20the%20forecasted%20data%20points)
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/CV_phase3final.jpg?raw=True)
# MAGIC
# MAGIC ### Modeling Environment
# MAGIC All modeling was completed in a DataBricks cluster with a standard_DS4_v2 28GB driver with 8 cores, and 2-6 standard_DS3_v2 workers with 28-84GB and 8-24 cores. In this environment, linear models took 1-10 minutes to train, while more complex MLP and graph neural networks took 12+ hours.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Summary of Model Experiments

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear Regression
# MAGIC
# MAGIC **Ordinary Least Squares (OLS) Linear Regression:** OLS linear regression is the most common version of linear regression. It estimates the relationship between independent variables and the dependent variable by minimizing the sum of the squared differences between the observed and predicted values. Using linear regression for this task has lower risk of overfitting, high model explainability, and allows for quick testing and iteration to optimize the modeling pipeline. We used linear regression to explore feature importance via model coefficients and guage the impact of techniques including:
# MAGIC
# MAGIC 1.  Addition of new, engineered features: The largest impact of a single engineered feature that we observed in linear models was 'PrevFlightDelay'.
# MAGIC 2.  Regularization: Minimal improvements seen with linear models.
# MAGIC 3. Downsampling: Minimal improvements seen with linear models, which is to be expected as downsampling is meant primarily for classification tasks.
# MAGIC
# MAGIC A key limitation of linear regression is that nonlinearity in the data is not captured through this approach. We moved on to other models that could represent nonlinear systems due to this limitation. 
# MAGIC
# MAGIC - **Implementation:** MLLib (`pyspark.ml.regression.LinearRegression`)
# MAGIC - **Number of Experiments Conducted:** 20+ 
# MAGIC - **Experimentation Details:** More modeling details can be seen in this [google sheet](https://docs.google.com/spreadsheets/d/1fcTjJKpwbI-GD0_ITRqAArBrXO3yETHPUMJImPrzJkk/edit?usp=sharing) and in the appendix. The best linear regression model on the 5 year dataset is below: 
# MAGIC
# MAGIC
# MAGIC Dataset | Model Summary | Average Cross Validation RMSE| Test RMSE | Baseline RMSE | Test F3 | Baseline F3 | Wall Time |
# MAGIC | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
# MAGIC | 5 year | Linear regression with all engineered features, capped departure delays as Y parameter, no regularization, no downsampling.  | 21.008 | 21.179 | 61.667 | 0.681 | 0.180 | 2 minutes |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-Layer Perceptron
# MAGIC
# MAGIC As part of the request for the client, we also looked to implement a Multi-Layer Perceptron (MLP) model to help us tackle the issue of delays. While we approached the larger request from a regression lens, we implemented a classification model due to technical challenges presented by the MLLib's package lack of support for MLP regression models. Nevertheless, we felt that exploring this model under a classification scope would still provide valuable insight such as understanding whether we are observing a non-linear relationship within our data. For this task, we explored several experiments with using an MLP and evaluated our models using F3 scores. Below are our findings:
# MAGIC
# MAGIC  | Models | Training Set Cross Validation F3 Score | Test F3 Score | Wall Time |
# MAGIC  |---|-----|---|---|
# MAGIC | 1 Hidden Layer with 2 Nodes | 0.781 | 0.758 | 1h |
# MAGIC | 1 Hidden Layer with 4 Nodes | 0.782 | 0.782 | 1h |
# MAGIC | 2 Hidden Layers (Hidden Layer 1 with 2 Nodes, Hidden Layer 2 with 2 Nodes) |  0.778 | 0.747 | 1h |
# MAGIC | 2 Hidden Layers (Hidden Layer 1 with 4 Nodes, Hidden Layer 2 with 2 Nodes) | 0.785 | 0.761 | 2h |
# MAGIC
# MAGIC | Baseline Model | F3 Score |
# MAGIC |--|--|
# MAGIC | Baseline Linear Regression Model | 0.180 |
# MAGIC
# MAGIC Our findings show that simplicity is key. The model that performed the best had only one hidden layer with 4 nodes. Our more complex models (i.e. 2 hidden layers) highlighted that greater number nodes moved the needle more than the number of layers. 
# MAGIC
# MAGIC In terms of limitation, we acknowledge that this model can't be compared to our other modeling work since we are working with two different tasks. However, the MLP model demonstrated a strong performance in classification and highlights the need for further investigation in finding other packages that can support an MLP regression task. Our other limitation is the lack of interpretability with these models. We can't explain which features distinctly have a greater effect on delays due to the black-box nature of these kind of models and, when working with business stakeholders, it presents a challenge to provide actionable insights due to this challenge. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### XGBoost
# MAGIC
# MAGIC The XGBoost model (SparkXGBRegressor) was implemented using time-series rolling cross-validation to predict flight delays. We conducted hyperparameter tuning to identify the best parameters for our model. The model was evaluated using RMSE and F3 scores.
# MAGIC
# MAGIC #### Pipeline
# MAGIC The XGBoost model pipeline included the following steps:
# MAGIC 1. Data preparation, including feature engineering and normalization.
# MAGIC 2. Hyperparameter tuning using grid search.
# MAGIC 3. Model training with the best parameters.
# MAGIC 4. Evaluation using RMSE and F3 metrics.
# MAGIC
# MAGIC #### Hyperparameter Tuning
# MAGIC Several configurations were tested, focusing on the maximum depth of the trees, learning rate, and the number of estimators. The results of these configurations are summarized in the table below.
# MAGIC
# MAGIC #### Results
# MAGIC The table below shows the average cross-validation RMSE, test RMSE, test F3 score, and baseline F3 score for different sets of hyperparameters. The best-performing model is highlighted.
# MAGIC
# MAGIC
# MAGIC | Model Parameters                | Avg. CV RMSE | Test RMSE | Test F3 | Baseline F3 | Wall Time | 
# MAGIC |-|--------------|-----------|---------|-------------|-|
# MAGIC | max_depth=4, learning_rate=0.2, n_estimators=150 | 17.31        | 15.95     | 0.791   | 0.180       | 2m |
# MAGIC | max_depth=6, learning_rate=0.2, n_estimators=200 | 16.60        | **14.96**     | **0.829**   | 0.180       | 2m |
# MAGIC | max_depth=6, learning_rate=0.2, n_estimators=150 | 18.23        | 15.61     | 0.703   | 0.180       | 2m |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Neural Network
# MAGIC
# MAGIC One fundemental issue that we noticed was that our model had no context about the rest of the world when it was trying to make a prediction. Events like Covid, CrowdStrike, Strikes, Hurricanes, etc. can cause large regions or even the entire country to experience delays simultaneously. In order to capture information about global delays, we experimented with creating a Graph Neural Network to create embeddings representing a global state.
# MAGIC
# MAGIC #### Graph Feature Engineering
# MAGIC
# MAGIC We started by grouping all flights into 2 hour chunks with the idea that the global state of the airline industry in the 2 hours before evaluating a model should be a fair representation about what delays are impacting the world at that time.
# MAGIC
# MAGIC For each time chunk, we created a bi-directional graph with nodes representing airports and edges representing flights between them. Each edge contained a vector of 8 values including weather and delay information for that flight. In the rare circumstance when there were multiple flights departing from the same origin to the same destination in the same time period, their values were averaged.
# MAGIC
# MAGIC The training set for the input features was in the shape of `(3546, 322, 322, 8)` - 3546 time periods, 322 airports, and 8 pieces of information per flight.
# MAGIC
# MAGIC #### Architecture
# MAGIC
# MAGIC The input to this network was the graph described in the previous section. The output was a vector of length 322 containing the average delay of flights leaving that airport 2 hours in the future.
# MAGIC
# MAGIC The tensorflow network architecture contained 6 graph convolution layers, 5 graph pooling layers, 1 dense intermediate layer, and 1 dense output layer.
# MAGIC
# MAGIC ```
# MAGIC _________________________________________________________________
# MAGIC  Layer (type)                Output Shape              Param #   
# MAGIC =================================================================
# MAGIC  conv2d_7 (Conv2D)           (None, 316, 316, 50)      19650     
# MAGIC                                                                  
# MAGIC  max_pooling2d_5 (MaxPoolin  (None, 158, 158, 50)      0         
# MAGIC  g2D)                                                            
# MAGIC                                                                  
# MAGIC  conv2d_8 (Conv2D)           (None, 152, 152, 50)      122550    
# MAGIC                                                                  
# MAGIC  max_pooling2d_6 (MaxPoolin  (None, 76, 76, 50)        0         
# MAGIC  g2D)                                                            
# MAGIC                                                                  
# MAGIC  conv2d_9 (Conv2D)           (None, 72, 72, 50)        62550     
# MAGIC                                                                  
# MAGIC  max_pooling2d_7 (MaxPoolin  (None, 36, 36, 50)        0         
# MAGIC  g2D)                                                            
# MAGIC                                                                  
# MAGIC  conv2d_10 (Conv2D)          (None, 32, 32, 50)        62550     
# MAGIC                                                                  
# MAGIC  max_pooling2d_8 (MaxPoolin  (None, 16, 16, 50)        0         
# MAGIC  g2D)                                                            
# MAGIC                                                                  
# MAGIC  conv2d_11 (Conv2D)          (None, 14, 14, 50)        22550     
# MAGIC                                                                  
# MAGIC  max_pooling2d_9 (MaxPoolin  (None, 7, 7, 50)          0         
# MAGIC  g2D)                                                            
# MAGIC                                                                  
# MAGIC  conv2d_12 (Conv2D)          (None, 5, 5, 50)          22550     
# MAGIC                                                                  
# MAGIC  flatten_1 (Flatten)         (None, 1250)              0         
# MAGIC                                                                  
# MAGIC  dense_2 (Dense)             (None, 500)               625500    
# MAGIC                                                                  
# MAGIC  dense_3 (Dense)             (None, 322)               161322    
# MAGIC                                                                  
# MAGIC =================================================================
# MAGIC Total params: 1099222 (4.19 MB)
# MAGIC Trainable params: 1099222 (4.19 MB)
# MAGIC Non-trainable params: 0 (0.00 Byte)
# MAGIC _________________________________________________________________
# MAGIC ```
# MAGIC
# MAGIC After training, the idea is to use this network as embeddings for the models that predict for an individual flight. Passing the global information through the network, and getting the intermediate representation of the 2nd to last layer. This should contain an abstract but information dense representation of the world at any given time.
# MAGIC
# MAGIC #### Issues
# MAGIC
# MAGIC We were unable to get this model trained within the Databricks environment. Since these types of models are not supported by the Spark ML library natively, we had used a library called Elephas to bridge Tensorflow to Spark. However, this library was not able to run successfully within the Databricks environment.
# MAGIC
# MAGIC Even if this experiment was able to train successfully, we had some doubts that it would improve our model's performance significantly. The graphs we had created were very sparse and high-dimensional which is a known difficult area of Machine Learning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and Discussion
# MAGIC
# MAGIC #### Results: Best Model and Pipeline Configuration
# MAGIC
# MAGIC Below we have listed the best model from each family and the XGBoost model has outperformed every other type of model while also running very efficiently.
# MAGIC
# MAGIC | Model | RMSE | F3 | Wall Time |
# MAGIC |-|-|-|-|
# MAGIC | Baseline | 61.667 | 0.180 | 1m |
# MAGIC | Linear Regression | 21.179 | 0.681 | 2m |
# MAGIC | XGBoost | **14.96** | **0.829** | 2m |
# MAGIC | Multi-Linear Preceptor | - | 0.782 | 1h |
# MAGIC
# MAGIC #### Discussion
# MAGIC
# MAGIC We were surprised to find that the MLP model was outperformed by the XGBoost as the MLP has the potential to capture far more complex patterns in the data. We believe that it failed to reach the same F3 score as XGBoost because it was trained on the classiciation problem by default instead of getting to use the higher precision target variable like the regression models did.
# MAGIC
# MAGIC #### Leakage
# MAGIC
# MAGIC Data leakage occurs when information from the validation/test sets is inadvertently used to train the model. This results in a model that performs better on the test data than it would in a real-world test because it has 'peeked at what's on the test'. The reported results for RMSE and F3 in the case of leakage become overly optimistic. In the context of time-series predictions, leakage of data from 'the future' is of particular concern. 
# MAGIC
# MAGIC Our team took pointed steps to eliminate the risk for data leakage at each step of our pipeline for the 5 year model. Sources of leakage and our strategies to avoid it are detailed below: 
# MAGIC
# MAGIC | ML Pipeline Step | Porential Source of Leakage | Mitigation |
# MAGIC | ----- | ----- | ----- |
# MAGIC | EDA | Inferences made on test data | EDA done only on train data |
# MAGIC | Data Cleaning & Preprocessing | Imputation | Imputation was not needed for our pipeline, but we would have imputed on train and test separately. |
# MAGIC | Data Cleaning & Preprocessing | Dropping cancelled flights| It is important to keep in mind this limits scope of model in certain performance environments. The performance predicting delays from ANY flight would be lower in reality due to cancelled flights. This needs to be addressed before deployment. |
# MAGIC | Feature Engineering | Previous Flight Delay | Only take previous flights > 2 hrs, generate separately for test/train. |
# MAGIC | Feature Engineering | Page Rank Feature | Page rank features generated separately for train and test sets. |
# MAGIC | Modeling Pipeline | Downsampling | While we didn't use it in our model pipeline, downsampling should be done only on train set, not for test or baseline. |
# MAGIC | Modeling Pipeline | Cross Validation | Rolling cross vlalidation splits for time series pose a higher risk of data leakage. Our team opted for blocked cross validation to reduce leakage. |
# MAGIC
# MAGIC #### Observations from Tuning Model Pipeline and Hyperparameters
# MAGIC
# MAGIC
# MAGIC **Observations on Feature Importance:** For each linear regression experiment conducted, we calculated the absolute value of the coefficients as a proxy for relative feature importance. Surprisingly, even after adding numerous thoughtfully engineered features, the top-ranking features were always from DEST or ORIGIN feature families. That is, the model is using the destination or origin airports to a large extent in it's prediction of a flight's delay time. This leads us to believe that some airports just can't get their act together! In order to account for this phenomenon, we added the page-rank parameter and implemented graph neural networks to model interdependencies between delays. Even with the pagerank features included, however, DEST and ORIGIN features always have the highest 'importance'. In a possible 'Phase 4' of this project, it would make sense to break airport operations down in deeper granularity. We suggest to include features that can better indicate whether a specific airport will be experiencing delays that day (i.e. logistics data such as number of planes on tarmac, staffing information, etc. ).
# MAGIC
# MAGIC To better gauge the relevance of other features, we tested models without the DEST and ORIGIN features included. The features with the highest and lowest magnitude coefficients are represented in the charts below. We can see that departure times in the afternoon and evening have high relative importance, which tracks with our EDA results. Previous flight delay is the sixth most high-ranking parameter, reaffirming our thought process when creating this feature. We also find that the carrier is important, suggesting that carrier operations in addition to airport operations would be relevant.
# MAGIC
# MAGIC Other features we created, however, did not rank as highly as we expected. Both pagerank features were among the lowest ranking features which merits further research into what features would better represent operational efficiency at airports. 
# MAGIC
# MAGIC 'is_holiday' was also low-ranking, but we know from experience how much holidays can impact airport travel plans. This warrants further review and possible scrutinty as to which types of holidays to flag (i.e Thanksgiving vs. Memorial Day).
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/top_feat_p3.png?raw=True)
# MAGIC
# MAGIC ![](https://github.com/darby-brown/airline_delays/blob/main/bottom_feat_p3.png?raw=True)
# MAGIC
# MAGIC
# MAGIC **Observations on Delay Departure Data Distrubution:** The target parameter, departure delays, had a left skewed distribution with a long tail. Many observations were equal to 0, yet some reached all the way past 2000 minutes. To mitigate the problem of non-normal distribution, we opted to cap the delay minutes, so all delays that exceeded a certain threshold were lumped into a single bucket. This change, alone, had a massive impact on performance, bringing our RMSE scores from 45-50 min range down to the 20 min range.
# MAGIC
# MAGIC **Observations on Downsampling:** When downsampling, we randomly eliminated 80% of observations with delay time of 0 minutes. This massive cut still left '0' as the majority delay time, and the downsampling attempt actually hurt model performance. Downsampling is typically a classification technique, so this makes sense. Based on the challenges we identified with the data distribution (left skew with too many zeros) and failed downsampling, we suggest as a next step to pair two models together to predict flight delays. The first model would be a classification model that predicts whether a flight will be on time, delayed, cancelled or diverted. The second model would be a regression model that would predict the time in minutes for **only** flights that are predicted to be delayed. This would likely result on a more reliable prediction of delay and delay time, with the ability to tune at each step with business-relevant metrics.
# MAGIC
# MAGIC **Time-Series Cross Validation:** The time-series cross validation models generally performed similarly to standard models, indicating that our model can be expected to be robust over time. 
# MAGIC
# MAGIC When comparing blocked to rolling cross validation, rolling splits performed markedly better than blocked. This may be because of the larger datasets that are used to train in rolling splits (and in this scenario, our dataset is limited to begin with). However, we opted to use blocked splits to reduce leakage.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC This project focused on predicting domestic flight delays for US airlines using machine learning, a critical task for improving passenger satisfaction and operational efficiency. Our hypothesis was that regression models incorporating custom features can serve as a good first step in estimating flight delays.
# MAGIC
# MAGIC We utilized historical flight data from the U.S. DOT and weather data from NOAA. We created features based on flight schedules, weather conditions, previous flight delays, and holidays, converting categorical data into one-hot encodings and normalizing numerical features. We developed and evaluated linear regression models with cross-validation, using RMSE and MAE metrics for evaluation. 
# MAGIC
# MAGIC Our models showed improvements over baseline predictions. For instance, our baseline model had an RMSE of 51 minutes. By adding engineered features, we reduced the test RMSE to 48 minutes. Further refinement, such as capping delay minutes and implementing cross-validation, brought the test RMSE of the XGBoost model down to 15 minutes
# MAGIC
# MAGIC Future work includes revisiting EDA and feature engineering, implementing RMSLE and MAPE metrics, and exploring nonlinear models like XGBoost and MLP Neural Networks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment Breakdown
# MAGIC
# MAGIC Development of an ML pipeline requires many tasks and interdependencies. Below are high-level estimations for work completed. Linked is a Gantt chart that details our high  execution plan and key milestones in more detail. [Link to Gantt Chart](https://docs.google.com/spreadsheets/d/1wVPY6MclSDAu62DsckMr3jBGuwGNDblA4pBhu7xhxHk/edit?usp=sharing)
# MAGIC
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
# MAGIC #### Phase 3
# MAGIC | Phase | Engineer | Description | Time Estimation |
# MAGIC | --------------------------------------------------- | ---------- | ------- | ------- |
# MAGIC | 3 | Abdul | XGB Modeling | 15 hour |
# MAGIC | 3 | Nick | MLP Modeling | 15 hour |
# MAGIC | 3 | Darby | Linear Regression Modeling | 4 hour |
# MAGIC | 3 | River | Graph Neural Network | 15 hour |
# MAGIC | 3 | Everyone | Final ML Pipeline | 10 hour |
# MAGIC | 3 | Everyone | Assembling Presentation | 4 hour |
# MAGIC | 3 | Darby, Everyone | Assembling Report | 10 hour |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix
# MAGIC
# MAGIC ### Notebooks:
# MAGIC Throughout the project, the team utilized several Databricks notebooks to document the different phases of data exploration, feature engineering, and model development.
# MAGIC
# MAGIC #### List of notebooks
# MAGIC
# MAGIC - **Final Report (Phase 3 - CalAir)**: The final comprehensive report documenting all the steps, analysis, and results from Phase 3 of the project.
# MAGIC - **Phase_3_EDA**: Exploratory Data Analysis (EDA) notebook where the initial data exploration and visualization were performed.
# MAGIC - **Phase_3_Feature_Engineering**: Notebook focused on feature engineering, including the creation and transformation of features to improve model performance.
# MAGIC - **Phase3_ML Pipeline_MLP_classification**: Notebook focused on implementing and tuning a Multi-Layer Perceptron (MLP) model for classification tasks.
# MAGIC - **Phase3_ML Pipeline_Regression_MLP**: Notebook dedicated to the development and tuning of an MLP model specifically for regression tasks.
# MAGIC - **Phase3_ML Pipeline_XGB**: Notebook detailing the XGBoost model pipeline, including hyperparameter tuning and time-series cross-validation.
# MAGIC - **Phase3_ML Pipeline_GNN**: Notebook that explores the implementation of Graph Neural Networks (GNN) within the context of the project.
# MAGIC
# MAGIC Notebook folder can be found here: 
# MAGIC https://adb-4248444930383559.19.azuredatabricks.net/browse/folders/3133664007179404?o=4248444930383559
# MAGIC
# MAGIC ### Additional Experiments
# MAGIC
# MAGIC In total, 20+ experiments were conducted for linear regression. Highlighted experiments can be seen in the below table. Details were summarized on a google sheet [here.](https://docs.google.com/spreadsheets/d/1fcTjJKpwbI-GD0_ITRqAArBrXO3yETHPUMJImPrzJkk/edit?usp=sharing) Additionally, models can be accessed and reviewed through MLFlow.
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