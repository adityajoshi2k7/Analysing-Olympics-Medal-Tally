import numpy as np
import pandas
#from subprocess import check_call
#import seaborn as sns 
#from matplotlib import pyplot as plt 
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV

def svm(X_train,y_train,X_test,y_test):
    print("1")
    X_train['Sex'],_ = pandas.factorize(X_train['Sex'])
    X_train['Sport'],_ = pandas.factorize(X_train['Sport'])
    X_train['Team'],_ = pandas.factorize(X_train['Team'])
    X_train['Host_Country'],_=pandas.factorize(X_train['Host_Country'])
    y_train['Medal'],_ = pandas.factorize(y_train['Medal'])
    X_test['Sex'],_ = pandas.factorize(X_test['Sex'])
    X_test['Sport'],_ = pandas.factorize(X_test['Sport'])
    X_test['Team'],_ = pandas.factorize(X_test['Team'])
    y_test['Medal'],_ = pandas.factorize(y_test['Medal'])
    X_test['Host_Country'],_=pandas.factorize(X_test['Host_Country'])
    print(X_test)
    print(X_test, y_test.values)
    grid_search(X_test, y_test)

def grid_search(X_train, Y_train):
    print("3")
    parameters = {'kernel':('linear', 'rbf','poly', 'sigmoid')}
            #'C':[1, 5, 10, 50, 100],
            #'gamma':['scale',0.25, .5], 'degree':[1, 2, 4]}
    print("4")
    svc = SVC()
    print("5")
    clf = GridSearchCV(svc, parameters, cv=2)
    print("6")
    print(len(X_train[:100]))
    clf.fit(X_train[:2500], Y_train[:2500].values.ravel())
    print("7")
    print(clf.best_params_)
    print("end")
