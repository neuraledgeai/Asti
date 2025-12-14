import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini AI Chat")

# 2. API Setup
# Retrieve API Key from Streamlit Secrets
try:
    API_KEY = st.secrets["asti_api_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

# Initialize the Client
client = genai.Client(api_key=API_KEY)

# 3. Initialize Session State (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
# We re-draw all previous messages every time the script re-runs
for message in st.session_state.messages:
    with st.spinner("Thinking... 🧠"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 5. Chat Input Listener
if prompt := st.chat_input("Ask Gemini something..."):
    
    # A. Display User Message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # B. Add User Message to Session State
    st.session_state.messages.append({"role": "user", "content": prompt})

    # C. Prepare History for Gemini API
    # Gemini expects roles to be "user" or "model" (Streamlit uses "assistant")
    # We convert the chat history to the format the SDK expects
    gemini_history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # D. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Note: Using the model you requested. 
            # If "gemini-2.5-flash" is not available yet, switch to "gemini-1.5-flash"
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful AI assistant."
                ),
                contents=gemini_history
            )
            
            bot_response = response.text
            message_placeholder.markdown(bot_response)
            
            # E. Add Assistant Message to Session State
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")
