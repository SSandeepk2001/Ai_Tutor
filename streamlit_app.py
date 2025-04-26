import streamlit as st
import joblib
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter

# Load the model
model = joblib.load(r"S:\Project\best_model_single.pkl")

# Page configuration
st.set_page_config(page_title="AI Tutor - IELTS Band Predictor", page_icon="🧠", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #1f3c88;
        text-align: center;
    }
    .stTextArea > label {
        font-weight: bold;
        color: #1f3c88;
        font-size: 18px;
    }
    .stButton button {
        background-color: #0077b6;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 12px 28px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #023e8a;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        margin-top: 3rem;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# App Title and Logo
st.image("https://cdn-icons-png.flaticon.com/512/3940/3940403.png", width=120)
st.title("🧠 AI Tutor: IELTS Essay Band Predictor")

st.markdown("""
Welcome to **AI Tutor**, your stylish and smart assistant for evaluating IELTS Writing Task 2 essays. ✍️

Paste your essay below and hit the **Predict Band Score** button. Your band prediction will appear with celebration! 🎉✨
""")

# Essay input
essay_input = st.text_area("📝 Enter Your IELTS Essay Below:", height=300, help="Paste your full IELTS Writing Task 2 essay here.")

# Prediction and Visualization section
if st.button("🚀 Predict Band Score"):
    if not essay_input.strip():
        st.warning("⚠️ Please enter an essay before predicting.")
    else:
        prediction = model.predict([essay_input])[0]
        st.success(f"🏆 Your Predicted IELTS Band Score is: **{prediction:.2f}** 🎯")
        st.balloons()

        # Visualizations
        st.markdown("### 📊 Essay Word Analysis")

        # Clean and tokenize essay
        words = re.findall(r'\b\w+\b', essay_input.lower())
        word_counts = Counter(words)

        # Word cloud
        wc = WordCloud(width=800, height=300, background_color='white').generate(' '.join(words))
        st.image(wc.to_array(), caption="Word Cloud of Your Essay")

        # Top word frequencies
        top_words = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Frequency'])
        st.bar_chart(top_words.set_index('Word'))

# Upload section
st.markdown("""
---
### 📁 Upload Your Essay File
You can also upload a `.txt` file containing your IELTS essay.
""")

uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("📜 Uploaded Essay Content:", value=content, height=300)
    if st.button("📈 Predict Uploaded Essay"):
        prediction = model.predict([content])[0]
        st.success(f"🏆 Predicted Band Score for Uploaded Essay: **{prediction:.2f}** 🎯")

# Footer
st.markdown("""
---
<div class="footer">Made with ❤️ by <strong>Sandeep Kumar</strong> | Powered by 🧠 Machine Learning</div>
""", unsafe_allow_html=True)
