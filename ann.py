import keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.datasets import make_classification
from numpy.random import seed
from tensorflow import set_random_seed

seed(10)
set_random_seed(20)
accuracy = []
max_accuracy = 0
optimial_params = []

def create_ann_model(activation = 'relu', neurons = 1, optimizer = 'adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim = 8, activation = activation))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = optimizer)
    return model

def ann_classifier(final_X, final_Y):
    # defining grid search parameters
    neurons = [2, 6, 8, 10, 14]      #[2, 4, 6, 8, 10]
    optimizer = ['adam']            #['adam', 'sgd', 'rmsprop']
    activation = ['relu']       #['relu', 'sigmoid', 'tanh', 'linear']
    epochs = [10]
    batch_size = [50]
    param_grid = dict(epochs = epochs, batch_size = batch_size, optimizer = optimizer, activation = activation, neurons = neurons)

    # Grid Search
    model = KerasClassifier(build_fn = create_ann_model)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1)
    grid_results = grid.fit(final_X, final_Y)

    # Best combination of hyper-parameters
    print('Best parameter: ', grid_results.best_score_, grid_results.best_params_)
