import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini AI Chat")
st.caption("Powered by Google Gemini 2.5 • Memory Optimized")

# 2. API Setup
try:
    API_KEY = st.secrets["asti_api_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

client = genai.Client(api_key=API_KEY)

# CONFIGURATION
# 1M tokens is huge, let's keep a safe buffer (e.g., 500k) or even smaller 
# for speed (e.g., 20k) depending on your needs.
MAX_CONTEXT_TOKENS = 500000 

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Helper Function: Convert Session State to Gemini Format
def get_gemini_history():
    gemini_history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    return gemini_history

# 5. Helper Function: Trim History if too long
def manage_token_limit():
    """Removes oldest messages if we exceed the limit."""
    history = get_gemini_history()
    
    try:
        # Ask Gemini how many tokens we are currently using
        count_info = client.models.count_tokens(
            model="gemini-2.5-flash",
            contents=history
        )
        current_tokens = count_info.total_tokens
        
        # While over limit, remove the oldest message (index 0)
        # We check > 1 to ensure we don't delete the very last message just sent
        while current_tokens > MAX_CONTEXT_TOKENS and len(st.session_state.messages) > 1:
            # Remove the oldest message from local state
            st.session_state.messages.pop(0)
            
            # Re-calculate tokens (simulated to save API calls, or real call)
            # For strict safety, we re-check API or estimate. 
            # Here we will re-generate history and check again in next loop.
            history = get_gemini_history()
            count_info = client.models.count_tokens(
                model="gemini-2.5-flash",
                contents=history
            )
            current_tokens = count_info.total_tokens
            
    except Exception as e:
        # If token counting fails, we just proceed to avoid breaking the app
        print(f"Token counting warning: {e}")

# 6. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Chat Input Listener
if prompt := st.chat_input("Ask Gemini something..."):
    
    # A. Display & Save User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Generate Response with Spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            
            # --- KEY CHANGE: MEMORY MANAGEMENT ---
            # Check and trim tokens before sending
            manage_token_limit()
            
            # Convert the (possibly trimmed) history for the API
            final_history = get_gemini_history()
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful AI assistant."
                    ),
                    contents=final_history
                )
                
                bot_response = response.text
                st.markdown(bot_response)
                
                # C. Save Assistant Message
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
