
# # Taxi drivers: How to earn money more efficiently?
# ### The taxi rides in Chicago (2016)
# This dataset includes taxi trips for 2016, reported to the City of Chicago in its role as a regulatory agency. Here, a subset of january data is analyzed in this notebook as a trial analysis. A direct question from this dataset that interests taxi driver is that what type of ride tends to have more earning? However, it basically depends on the taxi fare policy in Chicago. Another question more interesting is that **how to earn money more efficiently (i.e. higher fare / working time) for a taxi driver as an individual**. To answer this question, I grouped the dataset by taxi_id and generated new features for each driver, such as total rides, total working time (including the gap between billed trips), mean trip mile, most active community area, most active hour in the day, most active weekday and etc. A random forest regression model is applied to predict the working efficiency with a R^2 = 0.8534.
# 
# Data source: https://www.kaggle.com/chicago/chicago-taxi-rides-2016
# 
# Original data source: http://digital.cityofchicago.org/index.php/chicago-taxi-data-released



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pickle
import datetime
from __future__ import division
get_ipython().magic('matplotlib inline')

january = pd.read_csv('~\Downloads\chicago_taxi\chicago_taxi_trips_2016_01.csv', 
                      parse_dates=['trip_start_timestamp', 'trip_end_timestamp'])

# ## Impute missing values

# Nan value distribution
nrow = january.shape[0]
nans = january.isnull().sum()/nrow
nans = nans.sort_values(ascending=False)
print(nans)
fig, ax = plt.subplots(figsize = (10, 6))
nans.plot.barh()
plt.ylabel('Null value ratio')
plt.show()

# Drop columns with nan value proportion more than 40% 
january.drop(['pickup_census_tract', 'dropoff_census_tract'], axis=1, inplace=True)
# Drop rows (few) which contains nan value
january = january.dropna(subset=['trip_end_timestamp', 'fare', 'trip_seconds', 'taxi_id', 'trip_miles'])

def fillna_mean(df, col):
    # fillna cols in the list with col mean
    for c in col:
        mean = df[col].mean()
        df[col] = df[col].fillna(mean)

col_fillna_mean = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
fillna_mean(january, col_fillna_mean)
# fillna with -1 for the company column
january['company'] = january['company'].fillna(-1)

from sklearn.neighbors import KNeighborsClassifier

def fillna_community_area(df, col1, col2, col3):
    # Fill the nan value in community area with the latitude and longitude of closest neighbors 
    model = KNeighborsClassifier(n_neighbors=32)
    X = df.loc[~df[col3].isnull(), [col1, col2]]
    Y = df.loc[~df[col3].isnull(), col3]
    model.fit(X, Y)
    testx = df.loc[df[col3].isnull(), [col1, col2]]
    df.loc[df[col3].isnull(), col3] = model.predict(testx)
    
fillna_community_area(january, 'pickup_latitude', 'pickup_longitude', 'pickup_community_area')
fillna_community_area(january, 'dropoff_latitude', 'dropoff_longitude', 'dropoff_community_area')


january.head()


import pickle
# pickle precleaned dataset for future use 
january.to_pickle('201610')


jan = pd.read_pickle('201610')
nrow = jan.shape[0]
print(nrow)


# # Remove outlier and unreasonable entries


tip_pay = pd.DataFrame([jan['tips'] == 0, jan['payment_type'] == 'Cash'])
tip_pay = tip_pay.T
tip_pay.corr()


# The correlation matrix above indicates the payment_type (cash or credit card) is highly correlated with the result if tips is given. According to our real-life experience, it is probable that the tips given by cash is not recorded. Therefore, fare is used instead of trip total, to reflect more the reality.


# unreasonable: trip_seconds == 0 but earned more than base fare
print(((jan['trip_seconds'] == 0) & (jan['fare'] > 3.25)).sum())
jan = jan.drop(jan.index[(jan['trip_seconds'] == 0) & (jan['fare'] > 3.25)])
print(jan.shape)


# unreasonable: average speed trip_miles/ trip_seconds > 0.05 (180 mile/h)
print((jan.loc[jan['trip_seconds'] > 0, 'trip_miles'] / jan.loc[jan['trip_seconds'] > 0, 'trip_seconds']).describe())
print(((jan.loc[jan['trip_seconds'] > 0, 'trip_miles'] / jan.loc[jan['trip_seconds'] > 0, 'trip_seconds']) > 0.05).sum())
jan = jan.drop(jan.index[((jan['trip_seconds'] > 0) & ((jan['trip_miles'] / jan['trip_seconds']) > 0.05))])
print(jan.shape)


# unreasonable: charge much more than the Chicago taxi policy
# example: id: 7486, charge $85.00 fare with 3 min and 0 trip mile
print((jan['fare'] > 3.25 + 2 * (jan['trip_miles'] + 0.5) + 0.05 * jan['trip_seconds']).sum())
jan = jan.drop(jan.index[jan['fare'] > (3.25 + 2 * (jan['trip_miles'] + 0.5) + 0.05 * jan['trip_seconds'])])
print(jan.shape)


# unreasonable: no fare with either positive trip time or trip mile
print(((jan['fare'] == 0) & ((jan['trip_miles'] > 0) | (jan['trip_seconds'] > 0))).sum())
jan = jan.drop(jan.index[(jan['fare'] == 0) & ((jan['trip_miles'] > 0) | (jan['trip_seconds'] > 0))])
print(jan.shape)


# # Feature Engineering


jan['start_hour'] = jan['trip_start_timestamp'].dt.hour
jan['end_hour'] = jan['trip_end_timestamp'].dt.hour
jan['start_hour'].hist(alpha = 0.7, label=['pickup'], bins=24)
jan['end_hour'].hist(alpha = 0.2, label=['dropoff'], color='red', bins=24)
plt.xlabel('Hour')
plt.ylabel('Counts')
plt.show()


jan.groupby('start_hour')['fare'].mean().plot()
plt.ylabel('fare_per_ride')
plt.show()


jan['weekday'] = jan['trip_start_timestamp'].dt.weekday
jan.groupby('weekday')['fare'].mean().plot()
plt.ylabel('fare_per_ride')
plt.plot()


jan['trip_seconds'].describe()

jan.plot.scatter(x = 'trip_seconds', y = 'trip_miles')
plt.show()


# Rount contains both pickup_community_area and dropoff_community_area
jan['route'] = jan['pickup_community_area'].astype(int).astype(str) + '_' + jan['dropoff_community_area'].astype(int).astype(str)


# # Analyze taxi driver as an individual (groupby taxi_id) and create new features from those of rides

jan['taxi_id'].value_counts().hist()


taxi_id = jan.groupby('taxi_id')
# Total fare earned by each taxi driver in Jan 2016
print(taxi_id['fare'].sum().describe())
taxi_id['fare'].sum().sort_values(ascending=False).iloc[1:].hist()
plt.show()


# Create new df, recording taxi driver information
taxi_id2 = pd.DataFrame()
# Total fare earned by each taxi driver in Jan 2016
taxi_id2['total_fare'] = taxi_id['fare'].sum()
# Drop outlier (unreasonable earning)
taxi_id2 = taxi_id2.drop(np.argmax(taxi_id['fare'].sum()))
# Get company
taxi_id2['company'] = taxi_id['company'].agg(lambda x: x.value_counts().index[0])


jan[['taxi_id', 'company']].corr()


taxi_id2['company'].value_counts()


company_ct = taxi_id2['company'].value_counts() 
def group_company(x):
    if company_ct[x] < 50:
        return 0
    elif company_ct[x] < 150:
        return 1
    elif company_ct[x] < 1000:
        return 2
    elif company_ct[x] < 1500:
        return 3
    else:
        return x
taxi_id2['company'] = taxi_id2['company'].apply(lambda x: group_company(x))


# new features
taxi_id2['total_trip_time'] = taxi_id['trip_seconds'].sum()
taxi_id2['mean_trip_time'] = taxi_id['trip_seconds'].mean()
taxi_id2['trip_time_std'] = taxi_id['trip_seconds'].std()
taxi_id2['total_mile'] = taxi_id['trip_miles'].sum()
taxi_id2['mile_mean'] = taxi_id['trip_miles'].mean()
taxi_id2['mile_std'] = taxi_id['trip_miles'].std()
# total ride number
taxi_id2['trip_num'] = taxi_id.size()
# most probable hour to pickup
taxi_id2['max_hour'] = taxi_id['start_hour'].agg(lambda x: x.value_counts().index[0])
# least probable hour to pickup
taxi_id2['min_hour'] = taxi_id['start_hour'].agg(lambda x: x.value_counts().index[-1])
# most probable community area to pickup
taxi_id2['max_pickup_community'] = taxi_id['pickup_community_area'].agg(lambda x: x.value_counts().index[0])
# most probable community area to dropoff
taxi_id2['max_dropoff_community'] = taxi_id['dropoff_community_area'].agg(lambda x: x.value_counts().index[0])
# most probable route
taxi_id2['max_route'] = taxi_id['route'].agg(lambda x: x.value_counts().index[0])
# most probable weekday to pickup
taxi_id2['max_weekday'] = taxi_id['weekday'].agg(lambda x: x.value_counts().index[0])
# least probable weekday to pickup
taxi_id2['min_weekday'] = taxi_id['weekday'].agg(lambda x: x.value_counts().index[-1])


import datetime
def get_min(time_diff):
    return (time_diff.days * 1440 + time_diff.seconds // 60)

def get_work_time(x, first_date, last_date, total_min):
    # Calculate the real working time, assuming that drivers are not working for gap between trips > 2 hours 
    if x.shape[0] < 1:
        return 0    
    time_seq = x.sort_values(by = 'trip_start_timestamp')
    time_seq.reset_index(drop=True, inplace=True)
    nrow = time_seq.shape[0]
    if x.shape[0] == 1:
        return get_min(time_seq.loc[0, 'trip_end_timestamp'] - time_seq.loc[0, 'trip_start_timestamp'])
    before_work = get_min(time_seq.loc[0, 'trip_start_timestamp'] - first_date)
    after_work = get_min(last_date - time_seq.loc[nrow-1, 'trip_end_timestamp'])
    time_gap = time_seq['trip_start_timestamp'].shift(-1) - time_seq['trip_end_timestamp']
    time_in_min = time_gap[(time_seq['trip_start_timestamp'].shift(-1) > time_seq['trip_end_timestamp'])].apply(lambda x: get_min(x))
    time_in_min[-1] = 0
    total_rest = time_in_min[(time_in_min> 120)].sum()
    return (total_min - total_rest - before_work - after_work)

first_date = datetime.datetime.strptime('201601010000', "%Y%m%d%H%M")
last_date = datetime.datetime.strptime('201601312359', "%Y%m%d%H%M")
total_min = get_min(last_date - first_date)
print(total_min)
taxi_id2['work_time'] = taxi_id[['trip_start_timestamp', 'trip_end_timestamp']].                        apply(lambda x: get_work_time(x, first_date, last_date, total_min))

# To get a generalized conclusion, drivers working less than 24 hours per month is excluded 
taxi_id2 = taxi_id2.drop(taxi_id2.index[taxi_id2['work_time'] < 1440])
taxi_id2['trip_time_ratio'] = taxi_id2['total_trip_time'] / taxi_id2['work_time']
taxi_id2['fare_per_min'] = taxi_id2['total_fare'] / taxi_id2['work_time']
# Remove outlier
taxi_id2 = taxi_id2.drop(taxi_id2.index[taxi_id2['fare_per_min'] > 1.25])


# # Data Visualizaiton

# Is taxi driver working longer more effective? No.
sns.lmplot(data=taxi_id2, x = 'work_time', y = 'fare_per_min', fit_reg = True, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark"))
plt.show()


# Do you drive a lot to find a passenger?
sns.lmplot(data=taxi_id2, x = 'trip_time_ratio', y = 'fare_per_min', fit_reg = True, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark"))
plt.show()


# Long trip or short trip
sns.lmplot(data=taxi_id2, x = 'mile_mean', y = 'fare_per_min', fit_reg = False, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(145, 280, s=85, l=25, n=7))
plt.show()


# How flexible you are to trip miles
sns.lmplot(data=taxi_id2, x = 'mile_std', y = 'fare_per_min', fit_reg = False, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(145, 280, s=85, l=25, n=7))
plt.show()


sns.lmplot(data=taxi_id2, x = 'max_hour', y = 'fare_per_min', fit_reg = False, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(145, 280, s=85, l=25, n=7))
max_hour_fare = taxi_id2.groupby('max_hour')['fare_per_min'].mean()
max_hour_fare.plot(color = 'y')
plt.show()

sns.lmplot(data=taxi_id2, x = 'max_weekday', y = 'fare_per_min', fit_reg = False, hue= 'company', size = 10, 
           scatter_kws={'alpha':0.5}, palette = sns.diverging_palette(145, 280, s=85, l=25, n=7))
max_hour_fare = taxi_id2.groupby('max_weekday')['fare_per_min'].mean()
max_hour_fare.plot(color = 'y')
plt.show()


# Drop the 'total fare' and 'total_trip_time', which are closely related to the target and cannot be chosen by the driver.
taxi_id2.drop(['total_fare', 'total_trip_time'], axis = 1, inplace=True)
fig, ax = plt.subplots(figsize = (10, 8))
sns.heatmap(taxi_id2.corr(), ax = ax)
plt.show()


taxi_id2.corr()['fare_per_min']


taxi_id2.isnull().sum()


# # Random Forest Regressor 


fare_per_min = taxi_id2['fare_per_min']
taxi_id2.drop('fare_per_min', axis = 1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(taxi_id2, fare_per_min, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
model = RandomForestRegressor(n_estimators = 50)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mean_squared_error(pred, y_test))
print(r2_score(pred, y_test))


feature_importance = pd.DataFrame(model.feature_importances_)
feature_importance.index = taxi_id2.columns
feature_importance.columns = ['feature_importance']
feature_importance.sort_values(by='feature_importance', inplace=True)
print(feature_importance)
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance.plot.barh(ax=ax)
plt.show()




