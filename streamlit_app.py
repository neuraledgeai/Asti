import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini AI Chat")
st.caption("Powered by Google Gemini")

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

    # B. Prepare History for Gemini API (With Manual Truncation)
    gemini_history = []
    
    # Heuristic: 1 Token ~= 4 Chars. 
    # Max Limit is ~1M tokens. We set a safe buffer at ~800k tokens (3.2M chars).
    # You can lower this number (e.g., 100000) if you want the app to be faster.
    MAX_CHAR_LIMIT = 3200000 
    current_char_count = 0
    
    # We iterate backwards (newest to oldest) to ensure we keep the most recent context
    for msg in reversed(st.session_state.messages):
        msg_content = msg["content"]
        msg_len = len(msg_content)
        
        # Check if adding this message exceeds our safe limit
        if current_char_count + msg_len < MAX_CHAR_LIMIT:
            role = "user" if msg["role"] == "user" else "model"
            
            # Since we are iterating backwards, we insert at the beginning of the list [0]
            gemini_history.insert(0, 
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg_content)]
                )
            )
            current_char_count += msg_len
        else:
            # If limit is reached, stop adding older messages
            break

    # C. Generate Response with Spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            try:
                response = client.models.generate_content(
                    model="gemma-3-27b-it", # Ensure this model name matches your access level
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful AI assistant."
                    ),
                    contents=gemini_history # Sending the truncated history
                )
                bot_response = response.text
                st.markdown(bot_response)
                
                # D. Save Assistant Message to State
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
