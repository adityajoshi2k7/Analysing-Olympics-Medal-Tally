import numpy as np
import pandas
from subprocess import check_call
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from confusion_mat import conf_matrix
from yellowbrick.model_selection import ValidationCurve


# Decision Tree classifier
def decision_tree(final_X, final_Y, binary):
    parameters = {"criterion": ["gini","entropy"], 'max_depth':range(3,30)}
    dec_classifier = GridSearchCV(DecisionTreeClassifier(random_state=123), parameters, n_jobs=4, cv=10)
    dec_classifier.fit(X=final_X, y=final_Y)
    tree_model = dec_classifier.best_estimator_
    print ("best parameter>>>>>", dec_classifier.best_params_)
    export_graphviz(tree_model, out_file = 'figures/DecisionTree.dot', feature_names = final_X.columns)
    check_call(['dot','-Tpng','figures/DecisionTree.dot','-o','figures/DecisionTree.png'])

    y_pred = tree_model.predict(final_X)
    print('\nAccuracy: ', accuracy_score(final_Y, y_pred) * 100)
    precision, recall, fscore, support = precision_recall_fscore_support(final_Y, y_pred, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)
    conf_matrix(final_Y,y_pred, binary)
    plot_validation_curve(final_X,final_Y)

def plot_validation_curve(final_X, final_Y):

    viz = ValidationCurve(
        DecisionTreeClassifier(), param_name="max_depth",
        param_range=np.arange(1, 30), cv=10, scoring="accuracy"
    )
    viz.fit(final_X, final_Y)
    viz.poof()