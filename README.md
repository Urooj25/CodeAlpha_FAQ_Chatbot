🚀 AI FAQ Chatbot (NLP + Streamlit)
📌 Project Overview

This project is an AI-powered FAQ Chatbot built using Natural Language Processing (NLP) techniques. It intelligently understands user queries and provides the most relevant answers from a dataset.

The chatbot is deployed using Streamlit, providing a clean and interactive web interface.

🎯 Features

🤖 AI-based chatbot using NLP

🔍 Uses TF-IDF Vectorization for text representation

📊 Applies Cosine Similarity to find best matching answers

💬 Interactive web UI with Streamlit

🧠 Handles unknown queries using confidence threshold

📈 Displays confidence score for responses

📝 Maintains chat history



##Technologies Used

Python 🐍

Streamlit

Pandas

NLTK

Scikit-learn


##⚙️ How It Works

User enters a question

Text is preprocessed (lowercase, remove punctuation, tokenization)

TF-IDF converts text into numerical vectors

Cosine similarity compares user query with dataset questions

Best match is selected and returned as response

If similarity is low → chatbot gives fallback response


##📂 Project Structure

📁 AI-FAQ-Chatbot
│── app.py
│── faq_dataset.csv
│── README.md


##▶️ How to Run

Clone the repository:

git clone https://github.com/Urooj25/CodeAlpha_FAQ_Chatbot.git

Install dependencies:

pip install streamlit pandas nltk scikit-learn

Run the app:

streamlit run app.py


##💡 Future Improvements

Add voice input 🎤

Improve NLP using advanced models

Expand dataset for better accuracy

Deploy online (Streamlit Cloud / Render)



##🏆 Project Highlights

✔ Real AI (NLP-based, not rule-based)

✔ Interactive UI

✔ Beginner-friendly yet powerful

✔ Suitable for internships & competitions

👩‍💻 Author

Urooj Fatima
🔗 GitHub: https://github.com/Urooj25

⭐ Support

If you like this project, don’t forget to ⭐ the repo!
