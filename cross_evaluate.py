#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# File:                           cross_evaluate.py                 #
# Author:                         Aline Castendiek                  #
# Student ID:                     768297                            #
# Date:                           24/02/17                          #
# Operating system:               Mac OS X El Capitan [10.11.6]     #
# Python version:                 3.5.0                             #
# spaCy version:                  1.6.0                             #
# scikit-learn version:           0.18.1                            #
#                                                                   #
# This script performs cross-evaluation a specified data set.       #
# Also uses grid-search to find best hyperparameters for models.    #
#####################################################################

# Example call: python cross_evaluate.py

import random
import sys
import warnings
import numpy
from scipy import stats
from operator import itemgetter
from PrepareData import Preparation
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB   
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
    
    
# Path to annotated training data (will read in all .tsv files in the folder)
path_to_training_data = "training_data/*.tsv"    

    
#####################################################################
#               Cross Validation - Function                         #
#####################################################################

"""
Implements several different classifier models, performs a grid search for each of the models
to find its best hyperparameters and cross-evaluates the model using weighted F1 score as metric.

Returns model with the best F1 score. 

Note that since cv=10 is chosen as a parameter of 'cross_val_score', the 91 training examples will be split into 
ten subsets with only nine items each. One of the subsets will be used as training set, the other ones will be used to calculate the scores.
Since the splits of the data sets are randomized and there is such few data, the computed scores vary depending on the random split. This is also why there is no overall best model for this kind of data - the best resulting model depends on this random split.

"""
def cross_validation():

    trained_models = []
    
    clf_names = ["Decision Tree", "Bernoulli Naive Bayes", "Logistic Regression", "Random Forest",
                 "Linear SVM", "k-nearest Neighbors", "Perceptron"]

    # List of all classifiers to be tested
    classifiers_to_test = [
        DecisionTreeClassifier(),  
        BernoulliNB(),
        LogisticRegression(),
        RandomForestClassifier(),
        LinearSVC(),
        KNeighborsClassifier(),
        Perceptron()         
    ]
    
    # List of all parameter grids for the classifiers
    
    # Note: Some classifies have the optional parameter "class_weight" implemented. Using this parameter, one can produce
    # a prediction bias for a specific label so that the classifier will increase the probability of predicting this label. 
    # If a model offers this parameter, I use it to try out different weighting factors for the 'Reportivum' class during the
    # grid search.
    params_to_test = [
        
        {'max_depth': numpy.arange(3, 10),              # DecisionTreeClassifier param grid
         "class_weight" : [{'Reportivum': 1.5}, {'Reportivum': 2}, {'Reportivum': 3}, {'Reportivum': 3.5},'balanced']
        },  
        
        {'alpha':numpy.linspace(0.1,1,10)},             # BernoulliNB param grid
        
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],     # LogisticRegression param grid
          "class_weight" : [{'Reportivum': 1.5}, {'Reportivum': 2}, {'Reportivum': 3}, {'Reportivum': 3.5},'balanced']
        }, 
        
        {"n_estimators" : [250, 300, 400],              # RandomForestClassifier param grid
         "class_weight" : [{'Reportivum': 1.5}, {'Reportivum': 2}, {'Reportivum': 3}, {'Reportivum': 3.5},'balanced']
        },             
        
        {'fit_intercept': [True, False],                # LinearSVC param grid
         'loss': ["hinge", "squared_hinge"],
         'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
         'max_iter': [5, 10, 100, 1000],
         "class_weight" : [{'Reportivum': 1.5}, {'Reportivum': 2}, {'Reportivum': 3}, {'Reportivum': 3.5},'balanced']
        },
        
        {"n_neighbors": numpy.arange(1, 31, 2),        # KNeighborsClassifier param grid
        "metric": ["euclidean", "cityblock"],
        "weights": ["uniform", "distance"]},
        
        {'penalty':  ['l2', 'l1', 'elasticnet'],       # Perceptron param grid
         'alpha':numpy.linspace(0.1,1,10),
         'eta0':[0.8,0.9,1.1,1.2,1.3,1.5],
         "class_weight" : [{'Reportivum': 1.5}, {'Reportivum': 2}, {'Reportivum': 3}, {'Reportivum': 3.5},'balanced']
        }
    ]

    for clf, param_grid, name in zip(classifiers_to_test, params_to_test, clf_names): 
        
        print("Finding best hyperparameters for model:")
        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1_weighted', cv=10)
        grid_search.fit(X,y)
        
        print(grid_search.best_estimator_)
                
        print("Computing F1 Score for best model:")
        
        # Splits data, fits model and computes the scores 10 consecutive times (cv = 10; with different splits each time)
        # The tenth subset is used as a training set. Uses F1 Score as metric.
        scores = cross_val_score(grid_search.best_estimator_, X, y, cv=10, scoring="f1_weighted")
        
        print("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print('\n')
        
        trained_models.append((grid_search.best_estimator_,scores.mean(),name))

    # Return best model tuple
    return max(trained_models,key=itemgetter(1))


#####################################################################
#                         Read In Data                              #
#####################################################################

if len(sys.argv) == 1:
    
    texts, labels = Preparation.read_in_all_data(path_to_training_data)

    spacy_features = Preparation.collect_spaCy_data(texts)

    # Weights of this function can be tuned by hand: 
    feature_dicts = Preparation.create_spaCy_feature_dicts(spacy_features, 
                                                           STTS_WEIGHT=3,
                                                           POS_WEIGHT=2,
                                                           DEP_WEIGHT=3,
                                                           STTS_BIGRAM_WEIGHT=5,
                                                           POS_BIGRAM_WEIGHT=4,
                                                           DEP_BIGRAM_WEIGHT=5)
    # It is also possible to call the function without the extra parameters;
    # then all weights will have a default value of one.
    #feature_dicts = Preparation.create_spaCy_feature_dicts(spacy_features)

    training_data = Preparation.get_data_tuples(labels,feature_dicts)
    random.shuffle(training_data)

    # Unzip labels and texts into separate lists:
    labels, feature_vectors = zip(*training_data)


#####################################################################
#               Perform Cross Validation                            #
#####################################################################

    vectorizer = DictVectorizer()

    X = vectorizer.fit_transform(feature_vectors)
    y = labels 

    # Train a classifier
    print("Starting the cross-validation...\n")

    # Ignore warnings that one class cannot have less than ten training examples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Perform cross validation over different models with different parameters
        # and returns the model with the best score:
        best_model = cross_validation()

        print("Done!\n")    

        # Print the best model:
        print("The best found model with a weighted F1 score of %f:" % (best_model[1]))    
        print(best_model[0])

        # Optional: Print the n most discriminative features for every class (can be very useful for feature design)
        # Note that this will only work with linear models that have a coef_ class variable implemented:
        # Logistic Regression, SVM and Perceptron.
        # If the best model is one of the models above, the most discriminative features can be printed.
        if best_model[2] in ["Logistic Regression", "Linear SVM", "Perceptron"]:
            Preparation.print_most_informative_features(vectorizer, best_model[0], n=5)


####################################################################
#     Print Instructions                                           #
####################################################################

else:

    print("\nUSAGE:\n")
    print("python cross_evaluate.py\n\n")
    
    
