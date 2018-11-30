import numpy as np
import pandas
from sklearn.utils import resample


def sample_dataset():    
    original = pandas.read_csv('athlete_events.csv')
    city_country_map = {'Barcelona' : 'ESP', 'London' : 'GBR', 'Antwerpen' : 'BEL', 'Paris' : 'FRA', 'Los Angeles' : 'USA', 'Helsinki' : 'FIN', 'Sydney' :  'AUS', 'Atlanta' : 'USA', 'Stockholm' : 'SWE', 'Beijing' : 'CHN', 'Rio de Janeiro' : 'BRA',  'Athina' : 'GRE', 'Mexico City' : 'MEX', 'Munich' : 'GER', 'Seoul' : 'KOR', 'Berlin' : 'GER',  'Melbourne' : 'AUS', 'Roma' : 'ITA', 'Amsterdam' : 'NED', 'Montreal' : 'CAN', 'Moskva' : 'RUS', 'Tokyo' : 'JPN', 'St. Louis' : 'USA'}

    # remove winter entries and change host city to host country
    summer = original[~original.Season.str.contains("Winter")]
    summer = summer.drop(columns = ['Season', 'Games', 'Name', 'ID', 'Event','Team'])
    summer = summer.rename(index=str, columns={"City" : "Host_Country"})
    summer['Host_Country'] = summer['Host_Country'].replace(city_country_map)
    #print(summer)
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

    print(final_data['Medal'].value_counts())
    print('\nNull values per attribute: \n', final_data['Medal'].isnull().sum())

    # Separate majority and minority classes
    majority = final_data[final_data.Medal.isnull()]
    minority = final_data[final_data.Medal.notnull()]

    # Downsample majority class
    majority_downsampled = resample(majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=11000,     # to match minority class
                                    random_state=123) # reproducible results

    #Combine minority class with downsampled majority class
    final_data = pandas.concat([majority_downsampled,minority])
    #final_data=final_data[final_data.NOC=='USA']

    # Display new class counts
    print(final_data['Medal'].value_counts())
    print('\nNull values per attribute: \n', final_data['Medal'].isnull().sum())
    return final_data


