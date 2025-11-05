import streamlit as st

st.set_page_config(page_title="AI Fitness App", page_icon="💪", layout="centered")

st.title("💪 AI Fitness Trainer")
st.caption("Your personal AI-powered workout assistant")

st.markdown("""
### 🧭 Navigation
Use the sidebar to explore:
1️⃣ **📷 Live Stream** – real-time pose analysis  
2️⃣ **📤 Upload Video** – upload your workout for feedback  
3️⃣ **🤖 Chatbot** – get workout and diet guidance  
""")

st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=200)
st.markdown("---")
st.success("👈 Use the left sidebar to switch between pages.")
