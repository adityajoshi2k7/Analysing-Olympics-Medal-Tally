import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

def lstm_classifier(final_data):

	# country_count = len(final_data['NOC'].unique())
	# year_count = len(final_data['Year'].unique())

	# values = final_data.values
	# final_X = values[:, :-1]
	# final_Y = values[:, -1] 
	# print(country_count, ' ', year_count)

	# reshape - # countries, time series, # attributes (['Sex', 'Age', 'Height', 'Weight', 'NOC', 'Year', 'Host_Country', 'Sport'])
	#final_X = final_X.reshape(country_count, year_count, final_X.shape[1])
	final_X = final_data.groupby("NOC", as_index=True)['Year', 'Sex', 'Age', 'Height', 'Weight', 'Host_Country', 'Sport'].apply(lambda x: x.values.tolist())
	final_Y = final_data.groupby("NOC", as_index=True)['Medal'].apply(lambda x: x.values.tolist())
	final_X = pad_sequences(final_X, maxlen=None, dtype='float', padding='post', truncating='post', value=0.0)
	final_Y = pad_sequences(final_Y, maxlen=None, dtype='float', padding='post', truncating='post', value=0.0)

	print(final_X.shape)
	print(final_Y[189])
	print(final_Y.shape)

	# define model - 10 hidden nodes
	model = Sequential()
	model.add(LSTM(10, batch_input_shape = (2, len(final_X[0]), len(final_X[0][0])), return_sequences = True, stateful=True))
	model.add( LSTM(10, return_sequences = False))
	model.add(Dense(5272, activation = 'sigmoid'))
	model.summary()
	model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

	# fit network
	history = model.fit(final_X, final_Y, epochs = 10, batch_size = 2, verbose = 2)

	loss, accuracy = model.evaluate(final_X, final_Y)
	print(accuracy)
