import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page configuration
st.set_page_config(
    page_title="AI Spam Detection System",
    page_icon="🚨",
    layout="centered"
)

# Title
st.title("🚨 AI-Based Spam Detection System")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# Training data
data = {
    "message": [
        "Congratulations! You won a free iPhone",
        "You have won a lottery prize of 100000",
        "Claim your free cash prize now",
        "Click this link to win a free gift",
        "Urgent! You have won a reward",
        "Get free money by clicking this link",
        "You are selected for a free vacation",
        "Win a brand new car today",
        "Congratulations, claim your prize",
        "Free offer! Click now",
        
        "Hi, how are you?",
        "Are you coming to college today?",
        "Please call me when you are free",
        "I will reach home by 7 PM",
        "Can you send me the notes?",
        "Let's meet tomorrow",
        "Happy birthday! Have a great day",
        "What time is the class?",
        "Please submit the assignment today",
        "I am going to college now"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train AI model
model = MultinomialNB()
model.fit(X, y)

# Message input
message = st.text_area(
    "📩 Enter your message:",
    placeholder="Example: Congratulations! You won a free prize..."
)

# Prediction button
if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")

    else:
        # Convert input message
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Probability
        probabilities = model.predict_proba(message_vector)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE DETECTED!")
            st.write(f"Confidence: **{confidence:.2f}%**")

        else:
            st.success("✅ NOT A SPAM MESSAGE")
            st.write(f"Confidence: **{confidence:.2f}%**")

# Information section
st.markdown("---")
st.subheader("ℹ️ About the System")
st.write(
    "This AI-based system uses Natural Language Processing (NLP), "
    "TF-IDF feature extraction, and a Naive Bayes machine learning "
    "algorithm to classify messages as Spam or Not Spam."
)