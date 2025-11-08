import streamlit as st
import joblib
import re

model = joblib.load('models/sentiment_model2.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📝 Tweet Sentiment Analysis")
st.write("Predicts sentiment: Positive  | Neutral  | Negative ")

user_input = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        cleaned = clean_text(user_input)
        pred = model.predict([cleaned])[0]
        if pred == "positive":
            st.success("Sentiment: Positive :)")
        elif pred == "negative":
            st.error("Sentiment: Negative :( ")
        else:
            st.info("Sentiment: Neutral :| ")


