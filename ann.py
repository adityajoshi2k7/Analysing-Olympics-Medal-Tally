import keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


def create_ann_model(activation = 'relu', neurons = 1, optimizer = 'adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim = 8, activation = activation))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = optimizer)
    return model

def ann_classifier(final_X, final_Y):
    # defining grid search parameters
    neurons = [2, 4, 6, 8,10]      #[2, 4, 6, 8, 10]
    optimizer = ['adam','rmsprop']            #['adam', 'sgd', 'rmsprop']
    activation = ['relu', 'sigmoid', 'linear']       #['relu', 'sigmoid', 'tanh', 'linear']
    epochs = [10]
    batch_size = [500]
    param_grid = dict(epochs = epochs, batch_size = batch_size, optimizer = optimizer, activation = activation, neurons = neurons)

    # Grid Search
    model = KerasClassifier(build_fn = create_ann_model)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1 ,cv=10, verbose = 2)
    grid_results = grid.fit(final_X, final_Y)
    model = grid.best_estimator_

    # Best combination of hyper-parameters
    print('Best parameter: ', grid_results.best_score_, grid_results.best_params_)
    # params = grid_results.best_params_

    y_pred = model.predict(final_X)
    print('\nAccuracy: ', accuracy_score(final_Y, y_pred) * 100)
    precision, recall, fscore, support = precision_recall_fscore_support(final_Y, y_pred, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)