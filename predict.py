#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# File:                           predict.py                        #
# Author:                         Aline Castendiek                  #
# Student ID:                     768297                            #
# Date:                           01/02/17                          #
# Operating system:               Mac OS X El Capitan [10.11.6]     #
# Python version:                 3.5.0                             #
# spaCy version:                  1.6.0                             #
# scikit-learn version:           0.18.1                            #
#                                                                   #
# This script loads a stored trained model                          #
# and uses it to make predictions for unlabeled data.               #
#####################################################################

# Example call: python predict.py model.pickle

import pickle
import sys
from PrepareData import Preparation


# Path to unlabeled data (will read in all .tsv files in the folder)
path_to_unlabeled_data = "unlabeled_data/*.tsv"


#####################################################################
#              Load model and make predictions                      #
#####################################################################

"""
  Command line arguments: 
  [0]: predict.py
  [1]: name of file in which the trained model is saved

"""
if len(sys.argv) == 2:    

    # Load the trained model:
    clf, vectorizer = pickle.load(open(sys.argv[1], "rb"))
    
    # For unlabeled data the label list will simply look like this: [None, None, None, None...]
    texts, labels = Preparation.read_in_all_data(path_to_unlabeled_data)
    
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
    
    X_test = vectorizer.transform(feature_dicts)
    
    predicted_labels = clf.predict(X_test)
    
    # Print texts and corresponding predicted labels:
    for i in range(0,len(predicted_labels)):
        print(texts[i] + " -> " + predicted_labels[i])
        
    
####################################################################
#     Print Instructions                                           #
####################################################################

else:
    print("\nUSAGE:\n")
    print("python3 predict.py model.pickle\n")
    print("model.pickle: Saved pickle file containing the trained model.\n\n")
