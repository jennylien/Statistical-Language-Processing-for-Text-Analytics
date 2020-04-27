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
import string
import random
from nltk.corpus import stopwords


# Stemming and Lemmatization both generate the root form of the inflected words. 
# The difference is that stem might not be an actual word whereas, lemma is an actual language word.
# Lemmewq
lemmatizer = WordNetLemmatizer()


#initialzing Chatbot Training

words = []
documents = []
classes = []
punctuation = [x for x in string.punctuation]

data_file = open('intents.json').read()
intents_all = json.loads(data_file)

for intent in intents_all['intents']:
    for p in intent['patterns']:
        # tokenize each word  
        w = nltk.word_tokenize(p)
        words.extend(w)
        documents.append((w, intent['tag']))
        
        if intent['tag'] not in classes: 
            classes.append(intent['tag'])
            
# lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in punctuation]    

# set removes duplicates
words = sorted(list(set(words)))
classes = sorted(list(set(words)))



pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))







