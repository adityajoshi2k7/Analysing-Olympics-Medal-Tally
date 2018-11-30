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



# Encoding the attributes for classifier
def decision_tree(X_train,y_train,X_test,y_test):
    encoder = preprocessing.LabelEncoder()

    X_train['Sex'] = encoder.fit(X_train['Sex']).transform(X_train['Sex'])
    print('\nSex: ', encoder.classes_)
    X_train['Sport'] = encoder.fit(X_train['Sport']).transform(X_train['Sport'])
    print('\nSport: ', encoder.classes_)
    X_train['NOC']= encoder.fit(X_train['NOC']).transform(X_train['NOC'])
    print('\nNOC: ', encoder.classes_)
    X_train['Host_Country'] = encoder.fit(X_train['Host_Country']).transform(X_train['Host_Country'])
    print('\nHost_Country: ', encoder.classes_)
    y_train['Medal'] = encoder.fit(y_train['Medal']).transform(y_train['Medal'])

    X_test['Sex'] = encoder.fit(X_test['Sex']).transform(X_test['Sex'])
    X_test['Sport'] = encoder.fit(X_test['Sport']).transform(X_test['Sport'])
    X_test['NOC'] = encoder.fit(X_test['NOC']).transform(X_test['NOC'])
    X_test['Host_Country'] = encoder.fit(X_test['Host_Country']).transform(X_test['Host_Country'])
    y_test['Medal'] = encoder.fit(y_test['Medal']).transform(y_test['Medal'])

    #classifier('gini', X_train, y_train, X_test, y_test)
    classifier('entropy', X_train, y_train, X_test, y_test)


# def decision_tree(X_train,y_train,X_test,y_test):
#     X_train['Sex'],_ = pandas.factorize(X_train['Sex'])
#     X_train['Sport'],_ = pandas.factorize(X_train['Sport'])
#     X_train['NOC'],_ = pandas.factorize(X_train['NOC'])
#     X_train['Host_Country'],_=pandas.factorize(X_train['Host_Country'])	
    
#     y_train['Medal'],_ = pandas.factorize(y_train['Medal'])
    
#     X_test['Sex'],_ = pandas.factorize(X_test['Sex'])
#     X_test['Sport'],_ = pandas.factorize(X_test['Sport'])
#     X_test['NOC'],_ = pandas.factorize(X_test['NOC'])	
#     y_test['Medal'],_ = pandas.factorize(y_test['Medal'])
#     X_test['Host_Country'],_=pandas.factorize(X_test['Host_Country'])
#     classifier('gini',X_train,y_train,X_test,y_test)
#     classifier('entropy',X_train,y_train,X_test,y_test)


# Decision Tree classifier
def classifier(classifier,X_train,y_train,X_test,y_test):
    dec_classifier = DecisionTreeClassifier(criterion = classifier, random_state = 100, max_depth = 10)
    dec_classifier.fit(X_train, y_train)

    features = list(X_train.head(0))
    print(features)
    if classifier=='gini':
        export_graphviz(dec_classifier, out_file = 'figures/gini_tree.dot', feature_names = X_train.columns)
        check_call(['dot','-Tpng','figures/gini_tree.dot','-o','figures/gini_OutputFile.png'])
    else :
        export_graphviz(dec_classifier, out_file = 'figures/entropy_tree.dot', feature_names = X_train.columns)
        check_call(['dot','-Tpng','figures/entropy_tree.dot','-o','figures/entropy_OutputFile.png'])
        
    y_pred = dec_classifier.predict(X_test)
    print('\nClassifier :', classifier)
    print('\nAccuracy: ', accuracy_score(y_test, y_pred) * 100)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)

    #print(dec_classifier.predict_proba([[1, 25 ,173, 70, 204, 204, 20]]))
    
    class_names = ['No Medal','Medal']
    print(class_names)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print (classification_report(y_test, y_pred, target_names = class_names))
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.show()

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmp = cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmp)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
