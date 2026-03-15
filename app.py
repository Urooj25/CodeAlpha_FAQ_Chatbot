import streamlit as st
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# -----------------------------
# Load Dataset
# -----------------------------
try:
    data = pd.read_csv("faq_dataset.csv")
except:
    st.error("Dataset file not found. Please add faq_dataset.csv in this folder.")
    st.stop()

# -----------------------------
# Preprocess Function
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)

# Clean questions
data["clean_question"] = data["question"].apply(preprocess)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# -----------------------------
# Response Function
# -----------------------------
def get_response(user_question, threshold=0.3):
    clean_input = preprocess(user_question)
    user_vec = vectorizer.transform([clean_input])

    similarities = cosine_similarity(user_vec, X)
    max_sim = similarities.max()

    if max_sim < threshold:
        return "Sorry, I don't have an answer for that."

    idx = similarities.argmax()
    return data.iloc[idx]["answer"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 AI FAQ Chatbot")

st.write("Ask questions related to our FAQ database.")

user_input = st.text_input("Type your question here:")

if user_input:
    response = get_response(user_input)
    st.write("### Chatbot Response:")
    st.success(response)