
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK tokenizer
nltk.download('punkt')

# Load dataset
data = pd.read_csv("faq_dataset.csv")

# 1️⃣ Preprocess function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)

# 2️⃣ Clean questions
data["clean_question"] = data["question"].apply(preprocess)

# 3️⃣ TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# 4️⃣ Define get_response function **before using it**
def get_response(user_question, threshold=0.3):
    clean_input = preprocess(user_question)
    user_vec = vectorizer.transform([clean_input])
    similarities = cosine_similarity(user_vec, X)
    max_sim = similarities.max()
    
    if max_sim < threshold:
        return "Sorry, I’m not sure about that. 🤔"
    
    idx = similarities.argmax()
    return data.iloc[idx]["answer"]

# 5️⃣ Chat loop
print("🤖 Chatbot Ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! 👋")
        break
    response = get_response(user_input)  # ✅ This works now
    print("Chatbot:", response)