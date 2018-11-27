import numpy as np
import pandas
from subprocess import check_call
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# Encoding the attributes for classifier
def decision_tree(X_train,y_train,X_test,y_test):
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
    classifier('gini',X_train,y_train,X_test,y_test)
    classifier('entropy',X_train,y_train,X_test,y_test)

# Decision Tree classifier
def classifier(classifier,X_train,y_train,X_test,y_test):
    dec_classifier = DecisionTreeClassifier(criterion = classifier, random_state = 100, max_depth = 4)
    dec_classifier.fit(X_train, y_train)

    #features = list(X_train.head(0))
    if classifier=='gini':
        export_graphviz(dec_classifier, out_file = 'figures/gini_tree.dot', feature_names = X_train.columns)
        check_call(['dot','-Tpng','figures/gini_tree.dot','-o','figures/gini_OutputFile.png'])
    else :
        export_graphviz(dec_classifier, out_file = 'figures/entropy_tree.dot', feature_names = X_train.columns)
        check_call(['dot','-Tpng','figures/entropy_tree.dot','-o','figures/entropy_OutputFile.png'])
        
    y_pred = dec_classifier.predict(X_test)
    print('Classifier :', classifier)
    print('\nAccuracy: ', accuracy_score(y_test, y_pred) * 100)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)