#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####################################################################
# File:                           Preparation.py                    #
# Author:                         Aline Castendiek                  #
# Student ID:                     768297                            #
# Date:                           10/02/17                          #
# Operating system:               Mac OS X El Capitan [10.11.6]     #
# Python version:                 3.5.0                             #
# spaCy version:                  1.6.0                             #
# scikit-learn version:           0.18.1                            #
#                                                                   #
#   This file contains some functions that will be used             #
#   by the other scripts.                                           #
#####################################################################

import csv
import glob
import os
import spacy
import numpy

#####################################################################
#                      Read In Single File                          #
#####################################################################

# Read in data from a single file.
def read_in_single_file(file):

    tsv_reader = csv.reader(file, delimiter='\t')
    all_texts = []
    all_labels = []
    found_new_segment = False

    for row in tsv_reader:

        # Extract text of training example:
        if len(row) == 1 and row[0].startswith("#text="):
            found_new_segment = True
            all_texts.append(row[0].replace('#text=', ''))

            # Extract annotated label from first annotated row line
        if (len(row) > 0) and (not row[0].startswith("#")) and (found_new_segment == True):
            all_labels.append(get_label(row[2]))
            found_new_segment = False

    # The number of training examples and the number of labels must be the same size. If not, something went wrong!
    assert (len(all_texts) == len(all_labels)),"Number of training examples and corresponding labels must be the same!"

    return all_texts, all_labels


#####################################################################
#                            Get Label                              #
#####################################################################

# Extracts the label for the beginning of an annotated phrase.
def get_label(annotated_label):
    if annotated_label == "B-Reportivum": return "Reportivum"
    if annotated_label == "B-Estimativum": return "Estimativum"
    if annotated_label == "B-Evaluativum": return "Evaluativum"
    if annotated_label == "B-Direktivum": return "Direktivum"
    if annotated_label == "B-Hypoth-Sit": return "Hypoth-Sit"
    if annotated_label == "B-Meinung": return "Meinung"
    if annotated_label == 'O': return "**HEADLINE**"


#####################################################################
#                        Read In All Data                           #
#####################################################################

# Iterates over all files in a specified location and reads them in one by one.
def read_in_all_data(path_to_data):

    all_texts = []
    all_labels = []

    for current_path in glob.glob(path_to_data):
        head, tail = os.path.split(current_path)
        print("Reading in file: " + tail)

        with open(current_path, 'r') as current_file:
            texts, labels = read_in_single_file(current_file)

            # Remove headline from data (if there is one):
            if labels[0] == "**HEADLINE**":
                labels = labels[1:]
                texts = texts[1:]

            # The number of training examples and the number of labels must be the same size. If not, something went wrong!
            assert (len(texts) == len(labels)),"Number of training examples and corresponding labels must be the same!"

            # Add all texts and labels to complete list:
            for text in texts:
                all_texts.append(text)

            for label in labels:
                all_labels.append(label)

    return all_texts, all_labels


#####################################################################
#                         Collect SpaCy Data                        #
#####################################################################

# Collects features (STTS-tags, pos-tags, named-entity-types and dependency labels) via spaCy.
# Returns a list with one sublist for every training example.
# This sublist is a list of tuples where one tuple represents one token and its corresponding features, e.g.:
# [ ('Dagmar', 'NE', 'PROPN', 'PERSON', 'pnc'), ('Ziegler', 'NE', 'PROPN', 'PERSON', 'sb'), ... ]
def collect_spaCy_data(texts):
    
    print("Loading required spaCy model...")
    de_nlp = spacy.load('de')
    
    all_features = []
    
    # Iterate over texts:
    for i in range(0,len(texts)):
        # Analyze every training example via spaCy:
        analysis = de_nlp(texts[i])
        
        sentence_features = []
        # Iterate over found spaCy spans (can be more than one per training example):
        for sentence in analysis.sents: 
            # Iterate over span tokens:
            for token in sentence:
                # Collect token name, STTS-tag, pos-tag, named-entity-type and dependency label
                # for current token and save it as tuple:
                current_token_features = (token.orth_, token.tag_, token.pos_, token.ent_type_, token.dep_)
                sentence_features.append(current_token_features)
        all_features.append(sentence_features)

    # There must be one feature list for every training example. If not, something went wrong!
    assert(len(texts) == len(all_features)),"Number of feature lists and corresponding training examples must be the same!"
    
    return all_features


#####################################################################
#                      Create SpaCy Feature Dicts                   #
#####################################################################

"""
    Creates feature dicts from data.
    
    Following weights can be tuned by hand when calling this function:
    Single count weights:
    STTS_WEIGHT - Factor with which all single STTS tag counts will be multiplied.
    POS_WEIGHT - Factor with which all single POS tag counts will be multiplied.
    DEP_WEIGHT - Factor with which all single dependency label counts will be multiplied.
    Bigram weights:
    STTS_BIGRAM_WEIGHT - Factor with which all bigram STTS tag counts will be multiplied.
    POS_BIGRAM_WEIGHT - Factor with which all bigram STTS tag counts will be multiplied.
    DEP_BIGRAM_WEIGHT - Factor with which all bigram STTS tag counts will be multiplied.
"""
def create_spaCy_feature_dicts(feature_list, STTS_WEIGHT=1, POS_WEIGHT=1, DEP_WEIGHT=1, 
                               STTS_BIGRAM_WEIGHT=1, POS_BIGRAM_WEIGHT=1, DEP_BIGRAM_WEIGHT=1):

    # Create empty feature dicts (one for every training example):
    feature_dicts = [{} for i in range(0,len(feature_list))]

    for i in range(0,len(feature_list)):

        previous_token_tuple = ()

        for token_tuple in feature_list[i]:

            # Numerical features:

            # Create feature for occurring token and count them, e.g.: '**TOKEN=zu**': 1
            token_feature = "**TOKEN=" + str(token_tuple[0]) + "**"
            if token_feature not in feature_dicts[i]:
                feature_dicts[i][token_feature] = 1
            else: 
                feature_dicts[i][token_feature] += 1

            # Create feature for occurring STTS tag and count them, e.g.: '**STTS=NN**': 1
            tag_feature = "**STTS=" + str(token_tuple[1]) + "**"
            if tag_feature not in feature_dicts[i]:
                feature_dicts[i][tag_feature] = 1 * STTS_WEIGHT
            else: 
                feature_dicts[i][tag_feature] += 1 * STTS_WEIGHT

            # Create feature for occurring pos tag and count them, e.g.: '**POS=DET**': 1
            pos_feature = "**POS=" + str(token_tuple[2]) + "**"
            if pos_feature not in feature_dicts[i]:
                feature_dicts[i][pos_feature] = 1 * POS_WEIGHT
            else: 
                feature_dicts[i][pos_feature] += 1 * POS_WEIGHT

            # Collect entity information:
            if token_tuple[3] != '':
                # Create feature for occurring entity labels and count them, e.g.: '**ENT=PERSON**': 1
                feature_name = "**ENT=" + str(token_tuple[3]) + "**"

                if feature_name not in feature_dicts[i]:
                    feature_dicts[i][feature_name] = 1
                else:
                    feature_dicts[i][feature_name] += 1

            # Create feature for occurring dependency label and count them, e.g.: '**STTS=NN**': 1
            dep_feature = "**DEPENDENCY_LABEL=" + str(token_tuple[4]) + "**"
            if dep_feature not in feature_dicts[i]:
                feature_dicts[i][dep_feature] = 1 * DEP_WEIGHT
            else: 
                feature_dicts[i][dep_feature] += 1 * DEP_WEIGHT


            # Build simple bigram model for tokens, STTS tags, pos tags and dependency labels:
            if previous_token_tuple != ():

                # Token bigrams:
                token_bigram_feature = '**TOKEN[0]=' + previous_token_tuple[0] + ', TOKEN[1]=' + token_tuple[0] + '**'
                if token_bigram_feature not in feature_dicts[i]:
                    feature_dicts[i][token_bigram_feature] = 1
                else:
                    feature_dicts[i][token_bigram_feature] += 1

                # STTS tag bigrams:
                stts_bigram_feature = '**STTS[0]=' + previous_token_tuple[1] + ', STTS[1]=' + token_tuple[1] + '**'
                if stts_bigram_feature not in feature_dicts[i]:
                    feature_dicts[i][stts_bigram_feature] = 1 * STTS_BIGRAM_WEIGHT
                else:
                    feature_dicts[i][stts_bigram_feature] += 1 * STTS_BIGRAM_WEIGHT

               # pos tag bigrams:
                pos_bigram_feature = '**POS[0]=' + previous_token_tuple[2] + ', POS[1]=' + token_tuple[2] + '**'
                if pos_bigram_feature not in feature_dicts[i]:
                    feature_dicts[i][pos_bigram_feature] = 1 * POS_BIGRAM_WEIGHT
                else:
                    feature_dicts[i][pos_bigram_feature] += 1 * POS_BIGRAM_WEIGHT

                # dependency label bigrams:
                dep_bigram_feature = '**DEPENDENCY_LABEL[0]=' + previous_token_tuple[4] + ', DEPENDENCY_LABEL[1]=' + token_tuple[4] + '**'
                if dep_bigram_feature not in feature_dicts[i]:
                    feature_dicts[i][dep_bigram_feature] = 1 * DEP_BIGRAM_WEIGHT
                else:
                    feature_dicts[i][dep_bigram_feature] += 1 * DEP_BIGRAM_WEIGHT

            previous_token_tuple = token_tuple


            # Categorical features:

            # Dependency parse information:

            # Save root node token, STTS-tag and pos-tag:
            if token_tuple[4] == 'ROOT':
                feature_dicts[i]['**ROOT_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**ROOT_STTS**'] = token_tuple[1]
                feature_dicts[i]['**ROOT_POS**'] = token_tuple[2] 

            # Some dependency edge labels that could be particularly useful for identifying the training examples.
            # Save tokens, stts tags and pos tags for these specific edge labels as features.

            # cd = coordinating conjunction
            if token_tuple[4] == 'cd':
                feature_dicts[i]['**CD_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**CD_STTS**'] = token_tuple[1]
                feature_dicts[i]['**CD_POS**'] = token_tuple[2]

            # rc = relative clause 
            if token_tuple[4] == 'rc':
                feature_dicts[i]['**RC_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**RC_STTS**'] = token_tuple[1]
                feature_dicts[i]['**RC_POS**'] = token_tuple[2]

            # op = prepositional object
            if token_tuple[4] == 'op':
                feature_dicts[i]['**OP_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**OP_STTS**'] = token_tuple[1]
                feature_dicts[i]['**OP_POS**'] = token_tuple[2]

            # ju = junctor
            if token_tuple[4] == 'ju':
                feature_dicts[i]['**JU_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**JU_STTS**'] = token_tuple[1]
                feature_dicts[i]['**JU_POS**'] = token_tuple[2]

            # cp = complementizer
            if token_tuple[4] == 'cp':
                feature_dicts[i]['**CP_TOKEN**'] = token_tuple[0]
                feature_dicts[i]['**CP_STTS**'] = token_tuple[1]
                feature_dicts[i]['**CP_POS**'] = token_tuple[2]
            
    assert (len(feature_dicts) == len(feature_list)),"Number of feature dicts and corresponding lists must be the same!"                    
    return feature_dicts


#####################################################################
#                            Get Data Tuples                        #
#####################################################################

# Combines labels and feature dicts to tuples (to enable shuffling to randomize item order).
def get_data_tuples(labels, features):
    return [(labels[i],features[i]) for i in range(0,len(labels))]


#####################################################################
#               Print most informative features                     #
#####################################################################


"""
    Extracts the n most discriminative features from a classifier for every label. (For debugging and feature design purposes)

    Note that this will only work with linear models that have a coef_ class variable implemented.
    (Logistic Regression, SVM and Perceptron)
"""
def print_most_informative_features(vectorizer, clf, n=10):

    print("\nPrinting %d most important features for all occurring labels: \n" % (n))

    feature_names = vectorizer.get_feature_names()
    class_labels = clf.classes_
    
    for i, class_label in enumerate(class_labels):
        top_n = numpy.argsort(clf.coef_[i])[-n:]
        
        print(class_label + ": ")
        for j in top_n:
            print("%s" % (feature_names[j]))
        print('\n')

    