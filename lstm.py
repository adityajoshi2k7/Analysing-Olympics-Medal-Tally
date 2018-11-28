from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def lstm_classifier(final_data):

	values = final_data.values
	final_X = values[:, :-1]
	final_Y = values[:, -1] 

	final_X = final_X.reshape(len(final_X), 1, final_X.shape[1])

	# define model - 10 hidden nodes
	model = Sequential()
	model.add(LSTM(10, input_shape = (1, final_X.shape[1])))
	model.add(Dense(4, activation = 'sigmoid'))
	model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

	# fit network
	history = model.fit(final_X, final_Y, epochs = 10, batch_size = 50)

	loss, accuracy = model.evaluate(final_X, final_Y)
	print(accuracy)
