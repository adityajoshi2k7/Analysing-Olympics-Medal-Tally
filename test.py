#!/usr/bin/python
import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz

original = pandas.read_csv('athlete_events.csv')

# remove winter entries
summer = original[~original.Season.str.contains("Winter")]
summer = summer.drop(columns = ['Season', 'Games', 'Name', 'ID'])
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

# replace NA values with columnmean
final_data = final_data.fillna(final_data.mean())
print(final_data)

# Stratified Sampling - testing/training #214510 	#150154		#64356		
training_set = final_data[final_data['Year'] < 2000]
testing_set = final_data.drop(training_set.index, axis = 0)

# divide into X and y
X_train = training_set.drop('Medal', 1)
y_train = training_set['Medal']

X_test = testing_set.drop('Medal', 1)
y_test = testing_set['Medal']

gini_classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100)
gini_classifier.fit(X_train, y_train)

features = list(training_set.head(0))
#print(features)

def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


#visualize_tree(gini_classifier, features)