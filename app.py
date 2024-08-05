import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


## model the model file
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)


with open("vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

## streamlit app
st.title("Message Spam or Ham Classifier")
message = st.text_area("Type Message")

## cleaning preocess
corpus = []
words = re.sub("[^a-zA-Z]", " ", message)
words = words.lower()
words = words.split()
words = " ".join(words)
corpus.append(words)

## convert message into vector
vector = vectorizer.transform(corpus)

## train the model
prediction = model.predict(vector[0])

if st.button('Predict'):
    if prediction == 0:
        st.write("This is Ham Message !")
    else:
        st.write("This is Spam message")
