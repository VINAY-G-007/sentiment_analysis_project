import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 🎬 Load IMDb word index
# ---------------------------
word_index = keras.datasets.imdb.get_word_index()

# ---------------------------
# 🧠 Load trained LSTM model
# ---------------------------
# Load only the .keras model (new format)
model = keras.models.load_model("model/sentiment_model.keras", safe_mode=False)

# ---------------------------
# ⚙️ Constants
# ---------------------------
maxlen = 200

# ---------------------------
# 🔡 Function: Encode user input
# ---------------------------
def encode_text(text):
    words = text.lower().split()
    encoded = [1]  # 1 is the start token in IMDb dataset
    for word in words:
        encoded.append(word_index.get(word, 2))  # 2 for unknown words
    return pad_sequences([encoded], maxlen=maxlen)

# ---------------------------
# 💻 Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("🎭 Sentiment Analysis App")
st.write("This app uses an LSTM model trained on IMDb data to predict whether a text is **Positive** or **Negative**.")

# Input text box
user_input = st.text_area("Enter your review:", height=150)

# ---------------------------
# 🔍 Prediction Logic
# ---------------------------
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        encoded_input = encode_text(user_input)
        prediction = model.predict(encoded_input)[0][0]

        # Determine sentiment
        if prediction > 0.5:
            sentiment = "😊 Positive"
            confidence = prediction * 100
        else:
            sentiment = "😞 Negative"
            confidence = (1 - prediction) * 100

        # Display results
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")

# ---------------------------
# 🖤 Footer
# ---------------------------
st.markdown("---")
st.markdown("")
