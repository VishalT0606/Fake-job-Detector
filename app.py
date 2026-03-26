import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake Job Detector", page_icon="🔍")

st.title("🔍 Fake Job Detection System")
st.write("Enter job description to check if it's Fake or Real")

# Input
user_input = st.text_area("Paste job description here")

# Prediction function
def predict_job(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    
    if pred == 1:
        return "❌ Fake Job"
    else:
        return "✅ Real Job"

# Button
if st.button("Check Job"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_job(user_input)
        st.subheader(result)
