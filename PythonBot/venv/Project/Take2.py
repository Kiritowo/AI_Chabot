import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

from tensorflow.kears.models import load_model

lemmatizer = WordNetLemmatizer
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('Chatbot.h5')


def clean(sentence):
    input_words = nltk.word_tokenize(sentence)
    input_words = [lemmatizer.lemmatize(word) for word in input_words]
    return input_words


def bag_of_words(sentence):
    words_in_sentence = clean(sentence)
    bag = [0 for _ in range(len(words))]
    for w in words_in_sentence:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    margin_of_error = 0.1
    results = [[i, r] for i, r in enumerate(res)if r > margin_of_error]

    #if r > margin_of_error:
    #    results = [[i, r] for i, r in enumerate(res)]
    #else:
    #    print "Sorry I didn't understand that"

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list