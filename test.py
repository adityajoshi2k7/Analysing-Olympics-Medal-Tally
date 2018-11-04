#!/usr/bin/python
import numpy
import pandas

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

final_data = summer.loc[summer['Sport'].isin(recent_sports)]
print(final_data.loc[final_data.Sport == 'Badminton'])

# remove unwanted columns

null_columns=final_data.columns[final_data.isnull().any()]
print(final_data[null_columns].isnull().sum())
#