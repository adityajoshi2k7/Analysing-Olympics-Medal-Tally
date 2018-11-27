#!/usr/bin/python
import numpy as np
import pandas
from subprocess import check_call
import seaborn as sns
from matplotlib import pyplot as plt
from decisionTree import decision_tree
#from ann import ann_classifier

original = pandas.read_csv('athlete_events.csv')
city_country_map = {'Barcelona' : 'ESP', 'London' : 'GBR', 'Antwerpen' : 'BEL', 'Paris' : 'FRA', 'Los Angeles' : 'USA', 'Helsinki' : 'FIN', 'Sydney' :  'AUS', 'Atlanta' : 'USA', 'Stockholm' : 'SWE', 'Beijing' : 'CHN', 'Rio de Janeiro' : 'BRA',  'Athina' : 'GRE', 'Mexico City' : 'MEX', 'Munich' : 'GER', 'Seoul' : 'KOR', 'Berlin' : 'GER',  'Melbourne' : 'AUS', 'Roma' : 'ITA', 'Amsterdam' : 'NED', 'Montreal' : 'CAN', 'Moskva' : 'RUS', 'Tokyo' : 'JPN', 'St. Louis' : 'USA'}

# remove winter entries and change host city to host country
summer = original[~original.Season.str.contains("Winter")]
summer = summer.drop(columns = ['Season', 'Games', 'Name', 'ID', 'Event', 'NOC'])
summer = summer.rename(index=str, columns={"City" : "Host_Country"})
summer['Host_Country'] = summer['Host_Country'].replace(city_country_map)
print(summer)
latest_games = summer['Year'] > 2004
recent = summer[latest_games]

# find recent sports; remove sports not played now
recent_sports =  recent[['Sport', 'Year']].drop_duplicates().groupby('Sport')['Year'].count().reset_index()['Sport']
recent_sports = summer.loc[summer['Sport'].isin(recent_sports)]

# remove sports with insufficient data
recent_sports = recent_sports[['Sport', 'Year']].drop_duplicates().groupby("Sport")['Year'].count().reset_index()
recent_sports = recent_sports[recent_sports.Year > 6]['Sport']

# keep sports found in recent sports
final_data = summer.loc[summer['Sport'].isin(recent_sports)]

# Null and NA values - count
print('\nNull values per attribute: \n', final_data.isnull().sum())

# replace NA values with column mean
final_data['Height'].fillna((final_data['Height'].mean()), inplace = True)
final_data['Weight'].fillna((final_data['Weight'].mean()), inplace = True)
final_data['Age'].fillna((final_data['Age'].mean()), inplace = True)
print('\nCorrelation b/w Age, Height and Weight: \n', final_data[['Age', 'Height', 'Weight']].corr())
print('\n', final_data.describe())

# Medal Tally - Top 10 countries
medals_country = final_data.groupby(['Team','Medal'])['Sex'].count().reset_index().sort_values(by = 'Sex', ascending = False)
medals_country = medals_country.pivot('Team', 'Medal', 'Sex').fillna(0)
top = medals_country.sort_values(by = 'Gold', ascending = False)[:10]
top.plot.barh(width = 0.8, color=['#e78ae0', '#7eaee5', '#49ae7f'])
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.title('Medals Distribution Of Top 10 Countries')

# Distribution of Gold Medals vs Age
gold_medals = final_data[(final_data.Medal == 'Gold')]
sns.set(style = "darkgrid")
plt.tight_layout()
plt.figure(figsize = (20, 10))
sns.countplot(x = 'Age', data = gold_medals)
plt.title('Distribution of Gold Medals vs Age')

# Participation of Women
women_olympics = final_data[(final_data.Sex == 'F')]
plt.figure(figsize = (20, 10))
sns.countplot(x = 'Year', data = women_olympics)
plt.title('Women participation')
#plt.show()

# Men vs Women over time
men_dist = final_data[(final_data.Sex == 'M')]
men = men_dist.groupby('Year')['Sex'].value_counts()
women_dist = final_data[(final_data.Sex == 'F')]
women = women_dist.groupby('Year')['Sex'].value_counts()
plt.figure(figsize = (20, 10))
men.loc[:,'M'].plot()
women.loc[:,'F'].plot()
plt.legend(['Male', 'Female'], loc='upper left')
plt.title('Male and Female participation over the years ')
#plt.show()

# Indian Medals over the year
indian_medals = final_data[final_data.Team == 'India']
plt.figure(figsize = (20, 10))
plt.tight_layout()
sns.countplot(x = 'Year', hue = 'Medal', data = indian_medals)
plt.title("India's Total Medal count")
#plt.show()
# Stratified Sampling - testing/training #214510 	#150154		#64356		
training_set = final_data[final_data['Year'] < 2000]
testing_set = final_data.drop(training_set.index, axis = 0)
training_set=training_set.drop(columns = ['Year'])
testing_set=testing_set.drop(columns = ['Year'])

# divide into X and y
y_train = training_set[['Medal']].copy()
X_train = training_set.drop('Medal', 1)
y_train = y_train.replace(np.nan, 'No', regex = True)

X_test = testing_set.drop('Medal', 1)
y_test = testing_set[['Medal']].copy()
y_test = y_test.replace(np.nan, 'No', regex = True)


# Decision Tree Classifier
decision_tree(X_train,y_train,X_test,y_test)

# ANN Classifier
final_X = final_data.drop(columns = ['Medal'])
final_Y = final_data['Medal'] 
#ann_classifier(final_X, final_Y)


#for neuron in neurons :
#    for optimizer in optimizers :
#        for activation in activations :
#            bulid_ann_model (neuron, optimizer, X_train, y_train, activation)
#
#print(optimial_params, max_accuracy)








