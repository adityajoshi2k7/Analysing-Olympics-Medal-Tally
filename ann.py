import keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


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
    optimizer = ['adam', 'rmsprop']            #['adam', 'sgd', 'rmsprop']
    activation = ['relu', 'sigmoid', 'linear']       #['relu', 'sigmoid', 'tanh', 'linear']
    epochs = [10]
    batch_size = [1000]
    param_grid = dict(epochs = epochs, batch_size = batch_size, optimizer = optimizer, activation = activation, neurons = neurons)

    # Grid Search
    model = KerasClassifier(build_fn = create_ann_model)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, return_train_score = True, verbose = 2)
    grid_results = grid.fit(final_X, final_Y)

    # Best combination of hyper-parameters
    print('Best parameter: ', grid_results.best_score_, grid_results.best_params_)
    params = grid_results.best_params_

    X_train, X_test, y_train, y_test = train_test_split(final_X, final_Y, test_size = 0.25, random_state = 100)

    # create the model with the best params found
    best_model = create_ann_model(activation = params['activation'], 
        neurons = params['neurons'], optimizer = params['optimizer'])

    # Then train it and display the results
    history = best_model.fit(X_train, y_train, epochs = params['epochs'], batch_size = params['batch_size'])

    best_model.summary()
    plot_history(history)

    y_pred = best_model.predict_classes(X_test, batch_size = 50)
    print(classification_report(y_test, y_pred, target_names = [ "No Medal","Bronze","Gold", "Silver"]))
    

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
