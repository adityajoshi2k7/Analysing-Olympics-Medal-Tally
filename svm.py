import numpy as np
import pandas
#from subprocess import check_call
#import seaborn as sns 
#from matplotlib import pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV
from confusion_mat import conf_matrix

def svm(X_train, y_train, X_test, y_test):

    X_train['Sex'],_ = pandas.factorize(X_train['Sex'])
    X_train['Sport'],_ = pandas.factorize(X_train['Sport'])
    X_train['NOC'],_ = pandas.factorize(X_train['NOC'])
    X_train['Host_Country'],_=pandas.factorize(X_train['Host_Country'])
    y_train['Medal'],_ = pandas.factorize(y_train['Medal'])
    X_test['Sex'],_ = pandas.factorize(X_test['Sex'])
    X_test['Sport'],_ = pandas.factorize(X_test['Sport'])
    X_test['NOC'],_ = pandas.factorize(X_test['NOC'])
    y_test['Medal'],_ = pandas.factorize(y_test['Medal'])
    X_test['Host_Country'],_=pandas.factorize(X_test['Host_Country'])
    # params = grid_search(X_train, y_train.values.ravel())
    svclassification(X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
'''
def grid_search(X_train, Y_train):
    print("3")
    parameters = {'kernel':('linear', 'rbf','poly', 'sigmoid'),
            'C':[1, 5, 10, 50, 100],
            'gamma':['scale',0.25, .5], 'degree':[1, 2, 4]}
    print("4")
    svc = SVC()
    print("5")
    clf = GridSearchCV(svc, parameters)
    print("6")
    clf.fit(X_train[:100], Y_train[:100])
    print("7")
    print(clf.best_params_)
    print("end")
    return clf.best_params_['kernel']
'''
def svclassification(X_train, y_train, X_test, y_test):
    params = ['linear', 'sigmoid']
    accuracy_dict = {}
    for param in params:
        clf = SVC(kernel = param, class_weight='balanced', probability=True)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predict) * 100
        accuracy_dict[param] = accuracy
        print('\nAccuracy: ', accuracy)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, predict, average = 'micro')
        print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)
        conf_matrix(y_test,predict)
    # max_accuracy = max([accuracy_dict[param] for param in accuracy_dict])
    # #print("\n max accuracy: ",max_accuracy, " for kernel: ",str(param for param in accuracy_dict if accuracy_dict[param] == max_accuracy))
