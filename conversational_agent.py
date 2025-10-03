# Install dependencies
!pip install streamlit langchain faiss-cpu python-dotenv google-generativeai pyngrok langchain-google-genai langchain-community unstructured openpyxl

# Create environment file to store the API key
# Replace "your_google_gemini_api_key" with your actual key
import os
with open(".env", "w") as f:
    f.write("GOOGLE_API_KEY=YOUR_API_KEY")

# Import libraries and configure the API key
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Verify the list of available models
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)

# Code for the agent.py file
agent_code = """
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import asyncio

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=GOOGLE_API_KEY)

async def get_embeddings_async():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

_global_embeddings = None

def get_global_embeddings():
    global _global_embeddings
    if _global_embeddings is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            _global_embeddings = loop.run_until_complete(get_embeddings_async())
        else:
            _global_embeddings = asyncio.run(get_embeddings_async())
    return _global_embeddings

def load_vector_store(excel_path):
    embeddings = get_global_embeddings()
    loader = UnstructuredExcelLoader(excel_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    return vectordb

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain
"""

with open("agent.py", "w") as f:
    f.write(agent_code)

# Code for the updated app.py file
app_code = """
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM (ensure you have the correct model name)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# Define a prompt template that requests a direct recommendation
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="You are a 'Gift Genie' expert. Based on the user's prompt, provide a list of at least 5 creative gift recommendations. Do not ask for more information. Be concise and provide specific product types or ideas. User's prompt: {prompt}"
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# --- Streamlit UI Code ---

st.set_page_config(page_title="Gift Genie", layout="centered")
st.title("🎁 Gift Genie")
st.write("I'm here to help you find the perfect gift! Tell me who you're buying for and what they like.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("Tell me about the recipient (e.g., 'a boy who loves tech, 18 years old, budget 200'):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Conjuring up gifts..."):
        # Get response from the LLM based on the direct prompt
        response = chain.run(prompt=user_prompt)
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
"""

with open("app.py", "w") as f:
    f.write(app_code)

# Configure ngrok with your authtoken
# Replace "YOUR_NGROK_AUTHTOKEN" with your actual ngrok token
# If you don't have one, get it from https://dashboard.ngrok.com/get-started/your-authtoken
!ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN

# Run the streamlit app using ngrok tunnel
from pyngrok import ngrok
import threading
import time

def run_streamlit():
    os.system("streamlit run app.py &>/dev/null")

print("Starting Streamlit app...")
threading.Thread(target=run_streamlit).start()
time.sleep(5)

public_url = ngrok.connect(addr="8501", proto="http")
print(f"🚀 Your Gift Genie app is live at: {public_url}")
print("Click the URL above to open the app.")

