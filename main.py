#!/usr/bin/python
import numpy as np
import pandas
from subprocess import check_call
import seaborn as sns
from matplotlib import pyplot as plt
from decisionTree import decision_tree, plot_validation_curve
#from svm import svm

#from lstm import lstm_classifier
#from ann import ann_classifier

from sampling import sample_dataset
from sklearn.model_selection import train_test_split

binary, downsample = False, True
final_data = sample_dataset(downsample, binary)

# replace NA values with column mean
final_data['Height'].fillna((final_data['Height'].mean()), inplace = True)
final_data['Weight'].fillna((final_data['Weight'].mean()), inplace = True)
final_data['Age'].fillna((final_data['Age'].mean()), inplace = True)
'''
print('\nCorrelation b/w Age, Height and Weight: \n', final_data[['Age', 'Height', 'Weight']].corr())
print('\n', final_data.describe())


# Medal Tally - Top 10 countries
medals_country = final_data.groupby(['NOC','Medal'])['Sex'].count().reset_index().sort_values(by = 'Sex', ascending = False)
medals_country = medals_country.pivot('NOC', 'Medal', 'Sex').fillna(0)
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
indian_medals = final_data[final_data.NOC == 'IND']
plt.figure(figsize = (20, 10))
plt.tight_layout()
sns.countplot(x = 'Year', hue = 'Medal', data = indian_medals)
plt.title("India's Total Medal count")
#plt.show()
'''

training_set, testing_set = train_test_split(final_data, test_size = 0.25, random_state = 100)


# divide into X and y
y_train = training_set[['Medal']].copy()
X_train = training_set.drop('Medal', 1)
#y_train = y_train.replace(np.nan, 'No', regex = True)

X_test = testing_set.drop('Medal', 1)
y_test = testing_set[['Medal']].copy()
#y_test = y_test.replace(np.nan, 'No', regex = True)

final_X = final_data.drop(columns = ['Medal'])
final_Y = final_data['Medal']

# Decision Tree Classifier
decision_tree(final_X, final_Y, binary)


# ANN Classifier
 

#ann_classifier(final_X, final_Y)



# SVM Classifier
print("SVM Starting\n")
#svm(X_train, y_train, X_test, y_test)
print("SVM Completed\n")


#LSTM Classifier
#final_data.set_index('NOC', inplace = True)
#final_data.sort_index(inplace = True)
#final_data = final_data.groupby('NOC').groups
# final_data.sort_values('NOC', inplace = True)
# final_data = final_data.reset_index(drop = True)
# final_data.replace(np.nan, 'No', regex = True, inplace = True)
#print(final_data)
#lstm_classifier(final_data)


