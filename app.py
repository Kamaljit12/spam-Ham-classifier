import streamlit as st
import pandas as pd
import re
import pickle
from PIL import Image

from PIL import Image

def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)
    resized_image = image.resize((new_width, new_height))
    return resized_image

# Example usage:
resized_header_image = resize_image("spam-classifier.png", 300, 50)

# ------------------------------------

def main():
    st.title("My Streamlit App")
    st.image("spam-classifier.png", use_column_width=True)

if __name__ == "__main__":
    main()


## sample datasets
df = pd.read_csv('spam.csv', encoding='latin')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)

## show sample messages
sample_message = df.sample()
for lable in sample_message['label']:
    st.text(f"Message Type - {lable}")


for message in sample_message['message']:
    st.text(f"--------Copy and check the model with this message--------\n{message}")


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
