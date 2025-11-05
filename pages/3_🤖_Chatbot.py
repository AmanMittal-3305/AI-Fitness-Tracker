import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ----------------------------
# 🔧 Configure Gemini API key
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# ----------------------------
# ✅ Use a verified working model
# ----------------------------
MODEL_NAME = "models/gemini-2.5-flash"  # You can also try "models/gemini-2.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

# ----------------------------
# 🤖 Page Setup
# ----------------------------
st.set_page_config(page_title="AI Fitness Chatbot 🤖", page_icon="🤖", layout="centered")

st.title("🤖 AI Fitness Chatbot")
st.caption("Your virtual fitness assistant — ask me anything about workouts, form, or nutrition!")

# --- Load Robot Icon (Optional Fancy Header) ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)
with col2:
    st.markdown("### 💬 Chat with your virtual trainer")
    st.write("Type your question below and hit *Enter*!")

# ----------------------------
# 💬 Chat Setup
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me about workouts, diet, or exercise plans...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    try:
        with st.spinner("Thinking..."):
            response = model.generate_content(user_input)
            bot_reply = response.text
    except Exception as e:
        bot_reply = f"⚠️ Error: {str(e)}"

    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append({"role": "assistant", "text": bot_reply})

# ----------------------------
# 🗂️ Show chat history
# ----------------------------
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["text"])
