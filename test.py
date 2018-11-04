#!/usr/bin/python
import numpy
import pandas

df = pandas.read_csv('athlete_events.csv')

# remove winter entries
df = df[~df.Season.str.contains("Winter")]
#print(df)

sport_year =  df.groupby(['Sport', 'Year']).Year.count()

print(sport_year)
#['Year']['count'].sort_values() 			#.agg({'Year': ['count']})
#print(df.groupby(['Year', 'Team', 'Event', 'Medal']).agg('sum').reset_index())
#print(df.groupby(['Sport','Year']).count())