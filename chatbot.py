import streamlit as st
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Download tokenizer (only once)
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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
        return "I'm not fully sure about that yet 🤔 Try asking something else!", max_sim

    idx = similarities.argmax()
    return data.iloc[idx]["answer"], max_sim

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI FAQ Chatbot")
st.markdown("### 💬 Ask me anything about ML or general FAQs")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Type your question here:")

if user_input:
    response, score = get_response(user_input)

    # Save history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display chat history
for role, message in st.session_state.history:
    if role == "You":
        st.write(f"🧑 **You:** {message}")
    else:
        st.write(f"🤖 **Bot:** {message}")

# Show confidence (only for latest response)
if user_input:
    st.caption(f"Confidence Score: {score:.2f}")
