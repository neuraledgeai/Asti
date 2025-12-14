import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini AI Chat")
st.caption("Powered by Google Gemini 1.5 Flash")

# 2. API Setup
try:
    API_KEY = st.secrets["asti_api_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

client = genai.Client(api_key=API_KEY)

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input Listener
if prompt := st.chat_input("Ask Gemini something..."):
    
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Prepare History (Memory Management)
    # Instead of counting tokens, we simply keep the last 15 messages.
    # This prevents the "Context Limit" crash without any lag.
    recent_messages = st.session_state.messages[-15:]
    
    gemini_history = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # C. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            try:
                # Switched to 1.5-flash for better limits (1500/day vs 20/day)
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful AI assistant."
                    ),
                    contents=gemini_history
                )
                
                bot_response = response.text
                st.markdown(bot_response)
                
                # Save Assistant Message
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
