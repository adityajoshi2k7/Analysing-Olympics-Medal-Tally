#!/usr/bin/python
import numpy as np
import pandas
from subprocess import check_call
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

original = pandas.read_csv('athlete_events.csv')

# remove winter entries
summer = original[~original.Season.str.contains("Winter")]
summer = summer.drop(columns = ['Season', 'Games', 'Name', 'ID', 'Event', 'City'])
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

# replace NA values with column mean
final_data['Height'].fillna((final_data['Height'].mean()), inplace = True)
final_data['Weight'].fillna((final_data['Weight'].mean()), inplace = True)
final_data['Age'].fillna((final_data['Age'].mean()), inplace = True)

# Stratified Sampling - testing/training #214510 	#150154		#64356		
training_set = final_data[final_data['Year'] < 2000]
testing_set = final_data.drop(training_set.index, axis = 0)

# divide into X and y
y_train = training_set[['Medal']].copy()
X_train = training_set.drop('Medal', 1)
y_train = y_train.replace(np.nan, 'No', regex = True)

X_test = testing_set.drop('Medal', 1)
y_test = testing_set[['Medal']].copy()
y_test = y_test.replace(np.nan, 'No', regex = True)

# Encode string data for Decision Tree classifier
X_train['Sex'],_ = pandas.factorize(X_train['Sex'])
X_train['NOC'],_ = pandas.factorize(X_train['NOC'])
X_train['Sport'],_ = pandas.factorize(X_train['Sport'])
X_train['Team'],_ = pandas.factorize(X_train['Team'])	

y_train['Medal'],_ = pandas.factorize(y_train['Medal'])

X_test['Sex'],_ = pandas.factorize(X_test['Sex'])
X_test['NOC'],_ = pandas.factorize(X_test['NOC'])
X_test['Sport'],_ = pandas.factorize(X_test['Sport'])
X_test['Team'],_ = pandas.factorize(X_test['Team'])	

y_test['Medal'],_ = pandas.factorize(y_test['Medal'])

# Gini Classifier
def decision_tree(classifier):
	dec_classifier = DecisionTreeClassifier(criterion = classifier, random_state = 100, max_depth = 7)
	dec_classifier.fit(X_train, y_train)

	features = list(X_train.head(0))
	print(features)
	export_graphviz(dec_classifier, out_file = 'tree.dot')

	y_pred = dec_classifier.predict(X_test)
	print('\nAccuracy score for classifier: ', classifier, ' : ', accuracy_score(y_test, y_pred) * 100)

decision_tree('gini')
decision_tree('entropy')

