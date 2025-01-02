import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# # Ensure required NLTK data is available
# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transformed_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)


vectorizer_path = r'vectorizer.pkl'
model_path = r'model.pkl'


# Load files with error handling
try:
    tfidf =pickle.load(open(vectorizer_path, 'rb'))
except FileNotFoundError:
    st.error("The 'vectorizer.pkl' file was not found. Please ensure it exists in the script's directory.")
    st.stop()

try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error("The 'model.pkl' file was not found. Please ensure it exists in the script's directory.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")
if input_sms:
    # 1. Preprocess
    transformed_sms = transformed_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
