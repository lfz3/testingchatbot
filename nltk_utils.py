import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Lemmatize - option 1 - better words, less efficient
lemmatizer = WordNetLemmatizer()
def lemm(word):
    return lemmatizer.lemmatize(word.lower())

# Stemm - option 2 - more efficient than lemm, but results are sometimes not correct
stemmer = PorterStemmer()
def stemm(word):
    return stemmer.stem(word.lower())

# Remove stop words - e.g. 'this', 'are', 'is', etc.
stop_words = set(stopwords.words('english'))
def remove_stop_words(words):
    words = [word for word in words if word not in stop_words]
    return words

# Remove signs such as '!', '?', '.', etc.
def remove_punctuation(words):
    words = [word for word in words if word[0].isalpha()]
    return words

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [lemm(word) for word in tokenized_sentence]
    bag = numpy.zeros(len(words), dtype=numpy.float32)
    for index, word in enumerate(words):
        if word in tokenized_sentence:
            bag[index] = 1.0
    return bag
