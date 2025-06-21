import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake['label'] = 0
real['label'] = 1

texts = pd.concat([fake['title'], real['title']])
labels = pd.concat([fake['label'], real['label']])

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_news(news_text):
    input_data = vectorizer.transform([news_text])
    prediction = model.predict(input_data)
    return "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"

# ------------------ UI ------------------

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="centered",
)

# Theme toggle
mode = st.radio("🌓 Choose Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

# Apply custom background
if mode == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a headline and find out whether it's <b>Real</b> or <b>Fake</b> instantly.</p>", unsafe_allow_html=True)

# Input box
news_input = st.text_area("📝 Enter News Headline", height=100)

# Button
if st.button("🔍 Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        result = predict_news(news_input)
        if "Fake" in result:
            st.error(f"🔴 Prediction: {result}")
        else:
            st.success(f"🟢 Prediction: {result}")
