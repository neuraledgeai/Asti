import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import pypdf
import docx

# 1. Page Configuration
st.set_page_config(page_title="Gemini Pro Chat", page_icon="🤖", layout="wide")
st.title("🤖 Gemini AI Chat")
st.caption("Powered by Google Gemini 2.5 • Text, Images & Docs")

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

# --- HELPER FUNCTIONS FOR FILE PROCESSING ---
def get_pdf_text(uploaded_file):
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def get_docx_text(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading Word Doc: {e}")
        return None

# --- SIDEBAR: MULTIMODAL INPUTS ---
with st.sidebar:
    st.header("📂 Chat Options")
    st.write("Upload files to chat with them!")
    
    # Image Uploader
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    # Document Uploader
    uploaded_doc = st.file_uploader("Upload a Document", type=["pdf", "docx"])
    
    # Preview Image
    image_data = None
    if uploaded_image:
        image_data = Image.open(uploaded_image)
        st.image(image_data, caption="Image Preview", use_container_width=True)

    # Process Document
    doc_text = ""
    if uploaded_doc:
        if uploaded_doc.name.endswith(".pdf"):
            doc_text = get_pdf_text(uploaded_doc)
        elif uploaded_doc.name.endswith(".docx"):
            doc_text = get_docx_text(uploaded_doc)
        
        if doc_text:
            st.success(f"📄 {uploaded_doc.name} processed!")
            with st.expander("View extracted text"):
                st.write(doc_text[:500] + "...") # Preview first 500 chars

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display Text
        st.markdown(message["content"])
        # Display Image if present in that message
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)

# 5. Chat Input Listener
if prompt := st.chat_input("Ask Gemini something..."):
    
    # A. Construct the Full Prompt (User Input + Doc Context)
    full_prompt_text = prompt
    if doc_text:
        # We append the document text to the prompt invisibly to the context
        full_prompt_text = f"Context from uploaded document:\n{doc_text}\n\nUser Question: {prompt}"
    
    # B. Display User Message & Save to State
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_data:
            st.image(image_data, width=300)
    
    # Save to history. Note: We store the 'image_data' object if it exists.
    # We store 'full_prompt_text' (with doc context) so the model remembers the doc in future turns.
    st.session_state.messages.append({
        "role": "user", 
        "content": full_prompt_text, 
        "display_text": prompt, # Just for cleaner display if we wanted to separate them
        "image": image_data
    })

    # C. Prepare History for Gemini API (Sliding Window with Image Support)
    gemini_history = []
    
    MAX_CHAR_LIMIT = 3200000 
    IMAGE_TOKEN_COST = 3000 # Heuristic: Assume an image "costs" about 3000 chars (approx 800 tokens)
    current_char_count = 0
    
    for msg in reversed(st.session_state.messages):
        msg_content = msg["content"]
        msg_image = msg.get("image")
        
        # Calculate cost
        msg_cost = len(msg_content)
        if msg_image:
            msg_cost += IMAGE_TOKEN_COST
        
        if current_char_count + msg_cost < MAX_CHAR_LIMIT:
            role = "user" if msg["role"] == "user" else "model"
            
            # Create the Parts list
            parts = [types.Part.from_text(text=msg_content)]
            
            # If there is an image, add it to parts
            if msg_image:
                parts.append(types.Part.from_image(msg_image))
            
            gemini_history.insert(0, types.Content(role=role, parts=parts))
            current_char_count += msg_cost
        else:
            break

    # D. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", # Use 2.0 or 1.5-flash for best multimodal results
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful AI assistant. When analyzing documents or images, be detailed."
                    ),
                    contents=gemini_history
                )
                bot_response = response.text
                st.markdown(bot_response)
                
                # Save Assistant Message
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
