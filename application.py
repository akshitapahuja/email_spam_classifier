import streamlit as st
import joblib
import numpy as np

# Load saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.title("📨 Email Spam Classifier")
st.markdown("This app uses a Machine Learning model to classify whether an email is **Spam** or **Not Spam**.")

# Input box
email_text = st.text_area("✉️ Enter your email content below:", height=200)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Transform input text
        vectorized_input = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(vectorized_input)[0]
        proba = model.predict_proba(vectorized_input)[0]

        # Output
        if prediction == 1:
            st.error("🚫 This email is classified as **SPAM**.")
        else:
            st.success("✅ This email is classified as **NOT SPAM**.")

        # Show probabilities
        st.markdown(f"**Probability of Not Spam (0)**: `{proba[0]*100:.2f}%`")
        st.markdown(f"**Probability of Spam (1)**: `{proba[1]*100:.2f}%`")

# Footer
st.markdown("---")
