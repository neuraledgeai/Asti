import streamlit as st
from google import genai
from google.genai import types

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Gemini Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("💬 Gemini AI Chat")
st.caption("A tiny Streamlit + Gemini chat app")

# -----------------------------
# API KEY (direct paste, as requested)
# -----------------------------
asti_api_key = st.secrets["asti_api_key"]

client = genai.Client(api_key=asti_api_key)

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------
    # GEMINI RESPONSE
    # -----------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a helpful, concise AI assistant."
                    )
                ),
                contents=user_input
            )

            assistant_reply = response.text
            st.markdown(assistant_reply)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
