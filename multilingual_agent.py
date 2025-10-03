pip install googletrans==4.0.0-rc1 langdetect


from googletrans import Translator
translator = Translator()


# Install dependencies
!pip install streamlit langchain faiss-cpu python-dotenv google-generativeai pyngrok langchain-google-genai langchain-community unstructured openpyxl

# Create environment file to store the API key
# Replace "your_google_gemini_api_key" with your actual key
import os
with open(".env", "w") as f:
    f.write("GOOGLE_API_KEY=your_api_key")

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

# Clear the Streamlit resource cache to force re-indexing with the new code
!rm -rf /root/.streamlit/cache
print("✅ Streamlit cache cleared.")

!rm -rf /root/.streamlit/cache

import time 

pip install googletrans==4.0.0-rc1 langdetect




# =========================
# 🎁 Gift Genie - Full Setup
# =========================
# Run this in Google Colab or Jupyter Notebook

# Step 1: Install dependencies
print("📦 Installing dependencies...")
import subprocess, sys, os, time

packages = [
    "streamlit", "langchain", "langchain-google-genai", "langchain-community",
    "faiss-cpu", "python-dotenv", "google-generativeai", "pyngrok",
    "unstructured", "openpyxl", "googletrans==4.0.0-rc1", "langdetect"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("✅ Dependencies installed!")

# Step 2: Setup Google API Key
from getpass import getpass
api_key = getpass("Enter your Google API key: ")
with open(".env", "w") as f:
    f.write(f"GOOGLE_API_KEY={api_key}\n")
print("✅ API key stored in .env")

# Step 3: Clear Streamlit cache
os.system("rm -rf /root/.streamlit/cache")
print("✅ Streamlit cache cleared!")

# Step 4: Create agent.py
agent_code = """
import os, time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)
EMBEDDING_MODEL = "models/embedding-001"

def load_vector_store(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY, task_type="retrieval_document")
    loader = CSVLoader(file_path=file_path, encoding="ISO-8859-1")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)[:1000]
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5})
    QA_PROMPT = \"\"\"You are a helpful gift recommendation expert. Use the context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

Answer:\"\"\"
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": PromptTemplate.from_template(QA_PROMPT)}
    )
    return chain
"""

with open("agent.py", "w") as f:
    f.write(agent_code)
print("✅ agent.py created!")

# Step 5: Create app.py
app_code = """
import streamlit as st
import os, time
from dotenv import load_dotenv
from langdetect import detect
from googletrans import Translator

translator = Translator()

try:
    from agent import load_vector_store, build_qa_chain
except ImportError as e:
    st.error(f"Error importing agent: {e}")
    st.stop()

load_dotenv()
CSV_FILE_PATH = "flipkart_com-ecommerce_sample.csv"

st.set_page_config(page_title="Gift Genie", page_icon="🎁", layout="centered")
st.title("🎁 Gift Genie - Flipkart RAG")
st.markdown("### *AI-Powered Gift Recommendations from Flipkart*")
st.divider()

@st.cache_resource(show_spinner=False)
def setup_rag_chain(csv_path):
    if not os.path.exists(csv_path):
        st.warning("🚨 Upload 'flipkart_com-ecommerce_sample.csv' to the working directory")
        return None
    try:
        st.info("🔄 Initializing Gift Genie... This may take 3-5 minutes on first run")
        vectordb = load_vector_store(csv_path)
        qa_chain = build_qa_chain(vectordb)
        st.success("🎉 Gift Genie is ready!")
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error during setup: {str(e)}")
        return None

qa_chain = setup_rag_chain(CSV_FILE_PATH)
if qa_chain is None:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi! I'm Gift Genie. Tell me about the person you're shopping for, and I'll suggest perfect gifts from Flipkart's catalog!"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("💬 Describe who you're buying for (any language)..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        detected_lang = detect(user_prompt)
    except:
        detected_lang = "en"

    translated_prompt = translator.translate(user_prompt, src=detected_lang, dest="en").text if detected_lang != "en" else user_prompt

    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("🔍 Searching Flipkart products..."):
            try:
                result = qa_chain.invoke({"query": translated_prompt})
                response = result.get("result", "Sorry, I could not generate a response.")
                if detected_lang != "en" and response.strip() != "":
                    response = translator.translate(response, src="en", dest=detected_lang).text

                response_container.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                if "source_documents" in result and result["source_documents"]:
                    with st.expander("📦 View Retrieved Product Information"):
                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            st.markdown(f"**Source {i}:**")
                            content = doc.page_content
                            if len(content) > 500:
                                content = content[:500] + "..."
                            st.text(content)
                            if i < len(result["source_documents"][:3]):
                                st.divider()
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                response_container.error(error_msg)
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.markdown("### 🎁 Gift Genie")
    st.markdown("*Powered by AI & Flipkart Data*")
    st.divider()
    st.markdown("### 🔧 Technical Stack")
    st.markdown(\"\"\"
    - **LLM:** Google Gemini 2.0 Flash
    - **Embeddings:** Google Embedding-001
    - **Vector Store:** FAISS
    - **Framework:** LangChain
    - **Data Source:** Flipkart Products
    - **Translation:** Googletrans + Langdetect
    \"\"\")
    st.divider()
    if qa_chain:
        st.markdown("### 📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Status", "🟢 Active")
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")
"""

with open("app.py", "w") as f:
    f.write(app_code)
print("✅ app.py created!")

# Step 6: Launch Streamlit + ngrok
from pyngrok import ngrok
import threading

ngrok.set_auth_token("your_authtoken")  # Replace with your token

def run_streamlit():
    os.system("streamlit run app.py --server.port 8501 --server.headless true")

threading.Thread(target=run_streamlit, daemon=True).start()
time.sleep(10)

public_url = ngrok.connect(addr=8501, proto="http")
print(f"🎉 Your Gift Genie app is live: {public_url}")




# Configure ngrok with your authtoken
# Replace "YOUR_NGROK_AUTHTOKEN" with your actual ngrok token
# If you don't have one, get it from https://dashboard.ngrok.com/get-started/your-authtoken
!ngrok config add-authtoken your_authtoken

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

