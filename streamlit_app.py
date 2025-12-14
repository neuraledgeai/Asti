import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini AI Chat")
st.caption("Powered by Google Gemini 2.5")

# 2. API Setup
try:
    API_KEY = st.secrets["asti_api_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

client = genai.Client(api_key=API_KEY)

# 3. Initialize Session State (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input Listener
if prompt := st.chat_input("Ask Gemini something..."):
    
    # A. Display User Message & Save to State
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Prepare History for Gemini API
    # We convert the session state to the format Gemini expects
    gemini_history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # C. Generate Response with Spinner
    with st.chat_message("assistant"):
        # This is the new visual element you requested!
        with st.spinner("Thinking... 🧠"):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful AI assistant."
                    ),
                    contents=gemini_history # Sending full history for context
                )
                bot_response = response.text
                st.markdown(bot_response)
                
                # D. Save Assistant Message to State
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
