# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Jenny 
"""
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np 
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


# Stemming and Lemmatization both generate the root form of the inflected words. 
# The difference is that stem might not be an actual word whereas, lemma is an actual language word.
# Lemmewq
lemmatizer = WordNetLemmatizer()