from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def lstm_classifier(final_data):

	country_count = len(final_data['NOC'].unique())
	year_count = len(final_data['Year'].unique())

	values = final_data.values
	final_X = values[:, :-1]
	final_Y = values[:, -1] 
	print(country_count, ' ', year_count)

	# reshape - # countries, time series, # attributes
	final_X = final_X.reshape(country_count, year_count, final_X.shape[1])

	# define model - 10 hidden nodes
	model = Sequential()
	model.add(LSTM(10, input_shape = (country_count, final_X.shape[1])))
	model.add(Dense(4, activation = 'sigmoid'))
	model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

	# fit network
	history = model.fit(final_X, final_Y, epochs = 10, batch_size = 50)

	loss, accuracy = model.evaluate(final_X, final_Y)
	print(accuracy)
