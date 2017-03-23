#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# File:                           train.py                          #
# Author:                         Aline Castendiek                  #
# Student ID:                     768297                            #
# Date:                           01/02/17                          #
# Operating system:               Mac OS X El Capitan [10.11.6]     #
# Python version:                 3.5.0                             #
# spaCy version:                  1.6.0                             #
# scikit-learn version:           0.18.1                            #
#                                                                   #
# This script performs training on a specified data set.            #
# It saves the resulting model via pickle.                          #
#####################################################################

# Example call: python train.py model.pickle

import pickle
import sys
import random
from pprint import pprint
from PrepareData import Preparation
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier


# Path to annotated training data (will read in all .tsv files in the folder)
path_to_training_data = "training_data/*.tsv"


#####################################################################
#                            Training                               #
#####################################################################

"""
  Train a desired model.

  Command line arguments: 
  [0]: train.py
  [1]: name of file in which the trained model will be saved

"""
if len(sys.argv) == 2:        
    
    texts, labels = Preparation.read_in_all_data(path_to_training_data)

    spacy_features = Preparation.collect_spaCy_data(texts)
    #pprint(spacy_features[0])

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
   
    #pprint(feature_dicts[0])

    # Create label-feature tuples so that we can shuffle the data
    training_data = Preparation.get_data_tuples(labels,feature_dicts)
    random.shuffle(training_data)

    # Unzip labels and texts into separate lists
    labels, feature_vectors = zip(*training_data)

    # Transform data for sklearn
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(feature_vectors)
    y = labels

    # Train a classifier 
    # (I chose the best classifier I was able to find during cross-evaluation)
    print("Starting the training...")

    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cityblock',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='distance')
    clf.fit(X,  y)

    # Save trained model in separate file 
    print("Training done. Saving to file now...")
    pickle.dump((clf, vectorizer), open(sys.argv[1], "wb"))
    
    print("Done!")

    
####################################################################
#     Print Instructions                                           #
####################################################################

else:

    print("\nUSAGE:\n")
    print("python train.py model.pickle \n")
    print("model.pickle: Desired file in which the trained model will be saved via pickle.\n\n")
    
    