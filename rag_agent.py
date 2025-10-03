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

# Complete Fixed Gift Genie Code - Single File (WORKING VERSION)
# Run this in Google Colab or Jupyter Notebook

# ============================================
# STEP 1: Install Dependencies
# ============================================
print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    "streamlit",
    "langchain",
    "langchain-google-genai",
    "langchain-community",
    "faiss-cpu",
    "python-dotenv",
    "google-generativeai",
    "pyngrok",
    "unstructured",
    "openpyxl"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ All dependencies installed!")

# ============================================
# STEP 2: Setup Environment
# ============================================
import os
print("\n🔑 Setting up API key...")

# Create .env file with your API key
with open(".env", "w") as f:
    f.write("GOOGLE_API_KEY=your_api_key")

print("✅ API key configured!")

# ============================================
# STEP 3: Clear Cache
# ============================================
print("\n🧹 Clearing Streamlit cache...")
os.system("rm -rf /root/.streamlit/cache")
print("✅ Cache cleared!")

# ============================================
# STEP 4: Create agent.py (COMPLETELY FIXED VERSION)
# ============================================
print("\n📝 Creating agent.py with complete fix...")

agent_code = """
import os
import time
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

EMBEDDING_MODEL = "models/embedding-001"


def load_vector_store(file_path):
    \"\"\"
    COMPLETELY FIXED VERSION - Uses LangChain's native embedding directly
    No manual Google API calls that cause the 'content' parameter error
    \"\"\"
    print(f"\\n📊 Loading data from: {file_path}")
    
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Create embeddings using LangChain wrapper (handles API calls correctly)
    print("🔧 Initializing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )
    
    # Load and split documents
    print("📄 Loading CSV documents...")
    loader = CSVLoader(file_path=file_path, encoding="ISO-8859-1")
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents")
    
    print("✂️ Splitting documents into chunks...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = splitter.split_documents(documents)
    print(f"   Created {len(all_chunks)} text chunks")
    
    # Limit chunks for faster processing (optional)
    MAX_CHUNKS = 1000  # Process first 1000 chunks for demo
    if len(all_chunks) > MAX_CHUNKS:
        print(f"   ⚠️ Limiting to first {MAX_CHUNKS} chunks for faster setup")
        all_chunks = all_chunks[:MAX_CHUNKS]
    
    # Create vector store using LangChain's built-in method
    # This handles batching and API calls correctly
    print(f"\\n🔄 Creating embeddings for {len(all_chunks)} chunks...")
    print("   This will take several minutes. Please be patient...")
    
    try:
        # LangChain handles batching internally - no manual batching needed
        vectordb = FAISS.from_documents(
            documents=all_chunks,
            embedding=embeddings
        )
        
        print(f"\\n✅ Vector store created successfully!")
        print(f"   Total vectors: {vectordb.index.ntotal}")
        return vectordb
        
    except Exception as e:
        print(f"\\n❌ Error creating vector store: {e}")
        print("\\nTroubleshooting tips:")
        print("1. Check if API key is valid and has quota")
        print("2. Try reducing MAX_CHUNKS if hitting rate limits")
        print("3. Ensure stable internet connection")
        raise


def build_qa_chain(vectordb):
    \"\"\"
    Build the question-answering chain with custom prompt.
    \"\"\"
    print("\\n⛓️ Building QA chain...")
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 relevant documents
    )
    
    QA_PROMPT = \"\"\"You are a helpful gift recommendation expert. Your task is to use the provided Flipkart product data to suggest specific gifts.

CONTEXT (Product Data from Flipkart):
{context}

QUESTION: {question}

Instructions:
1. Analyze the context carefully to find relevant products
2. Suggest at least 3 specific products with names and prices if available
3. If no specific products match, provide general category suggestions
4. Be helpful and specific in your recommendations

Answer:\"\"\"
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(QA_PROMPT)
        }
    )
    
    print("✅ QA chain ready!\\n")
    return chain
"""

with open("agent.py", "w") as f:
    f.write(agent_code)

print("✅ agent.py created successfully!")

# ============================================
# STEP 5: Create app.py (Streamlit UI)
# ============================================
print("\n📝 Creating app.py...")

app_code = """
import streamlit as st
import os
import time
import sys
from dotenv import load_dotenv

# Import our agent functions
try:
    from agent import load_vector_store, build_qa_chain
except ImportError as e:
    st.error(f"Error importing agent: {e}")
    st.stop()

# Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSV_FILE_PATH = "flipkart_com-ecommerce_sample.csv"

# Page config
st.set_page_config(
    page_title="Gift Genie - Flipkart RAG",
    page_icon="🎁",
    layout="centered"
)

# Custom CSS for better styling
st.markdown(\"\"\"
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
\"\"\", unsafe_allow_html=True)

# Setup RAG chain (cached for performance)
@st.cache_resource(show_spinner=False)
def setup_rag_chain(csv_path):
    \"\"\"Initialize the RAG system with error handling.\"\"\"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        st.error(f"🚨 Data file not found: {csv_path}")
        st.info("📥 Please upload 'flipkart_com-ecommerce_sample.csv' to the working directory.")
        st.markdown(\"\"\"
        **Steps to fix:**
        1. Download the Flipkart dataset
        2. Upload it to the same directory as this app
        3. Refresh the page
        \"\"\")
        return None
    
    try:
        # Create progress container
        progress_container = st.empty()
        
        with progress_container.container():
            st.info("🔄 **Initializing Gift Genie...**")
            st.markdown("*This process may take 3-5 minutes on first run*")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load vector store
            status_text.text("📊 Loading and processing data...")
            progress_bar.progress(20)
            
            vectordb = load_vector_store(csv_path)
            progress_bar.progress(70)
            
            # Step 2: Build QA chain
            status_text.text("⛓️ Building question-answering chain...")
            qa_chain = build_qa_chain(vectordb)
            progress_bar.progress(100)
            
            status_text.text("✅ Setup complete!")
            time.sleep(1)
        
        progress_container.empty()
        st.success("🎉 **Gift Genie is ready!** Powered by Flipkart product data.")
        
        return qa_chain
        
    except Exception as e:
        st.error(f"❌ **Error during setup**")
        st.exception(e)
        
        with st.expander("🔍 See detailed error information"):
            st.code(str(e))
            st.markdown(\"\"\"
            **Common issues:**
            - API key expired or invalid
            - Rate limit exceeded (wait a few minutes)
            - Network connectivity issues
            - CSV file corrupted or wrong format
            \"\"\")
        
        return None

# Initialize the system
qa_chain = setup_rag_chain(CSV_FILE_PATH)

# UI Header
st.title("🎁 Gift Genie")
st.markdown("### *AI-Powered Gift Recommendations from Flipkart*")
st.divider()

if qa_chain is None:
    st.warning("⚠️ **RAG system failed to initialize.** Please check the error above and try again.")
    st.stop()

# Info section
with st.expander("ℹ️ How to use Gift Genie", expanded=False):
    st.markdown(\"\"\"
    **Gift Genie helps you find perfect gifts from real Flipkart products!**
    
    **Example queries:**
    - *"Gift for a tech-loving 18-year-old boy, budget under ₹5000"*
    - *"Birthday gift for a woman who loves fashion"*
    - *"Anniversary gift ideas under ₹10000"*
    - *"Gift for a gamer teenager"*
    - *"Kitchen appliances for a new home"*
    
    **Tips:**
    - Be specific about the person and occasion
    - Mention budget if you have one
    - Include interests or hobbies
    \"\"\")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hi! I'm Gift Genie. Tell me about the person you're shopping for, and I'll suggest perfect gifts from Flipkart's catalog!"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_prompt := st.chat_input("💬 Describe who you're buying for..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        with st.spinner("🔍 Searching through Flipkart products..."):
            try:
                # Query the RAG chain
                result = qa_chain.invoke({"query": user_prompt})
                response = result.get('result', 'Sorry, I could not generate a response.')
                
                # Display response
                response_container.markdown(response)
                
                # Show source documents
                if 'source_documents' in result and result['source_documents']:
                    with st.expander("📦 View Retrieved Product Information"):
                        for i, doc in enumerate(result['source_documents'][:3], 1):
                            st.markdown(f"**Source {i}:**")
                            content = doc.page_content
                            # Truncate if too long
                            if len(content) > 500:
                                content = content[:500] + "..."
                            st.text(content)
                            if i < len(result['source_documents'][:3]):
                                st.divider()
                else:
                    with st.expander("ℹ️ Note"):
                        st.info("No specific product data was retrieved for this query.")
                
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                response_container.error(error_msg)
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
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
    \"\"\")
    
    st.divider()
    
    # Stats
    if qa_chain:
        st.markdown("### 📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Status", "🟢 Active")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Info
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created successfully!")

# ============================================
# STEP 6: Setup ngrok and Launch
# ============================================
print("\n🌐 Setting up ngrok tunnel...")

# Configure ngrok
os.system("ngrok config add-authtoken your_authtoken > /dev/null 2>&1")

print("✅ ngrok configured!")

# ============================================
# STEP 7: Launch Application
# ============================================
print("\n🚀 Launching Gift Genie application...")
print("=" * 60)

import threading
import time
from pyngrok import ngrok

def run_streamlit():
    """Run Streamlit in background."""
    os.system("streamlit run app.py --server.port 8501 --server.headless true > /dev/null 2>&1")

# Start Streamlit in separate thread
print("Starting Streamlit server...")
threading.Thread(target=run_streamlit, daemon=True).start()

# Wait for server to start
print("Waiting for server to initialize...")
time.sleep(10)

# Create ngrok tunnel
try:
    public_url = ngrok.connect(addr="8501", proto="http", bind_tls=True)
    
    print("\n" + "=" * 60)
    print("✨ SUCCESS! Your Gift Genie app is now live!")
    print("=" * 60)
    print(f"\n🔗 Access your app here: {public_url}")
    print("\n📝 Important Notes:")
    print("   • First load will take 3-5 minutes to index data")
    print("   • Make sure 'flipkart_com-ecommerce_sample.csv' is uploaded")
    print("   • Keep this notebook running to keep the app alive")
    print("   • The app uses Google Gemini API (check your quota)")
    print("\n💡 Tip: If you get errors, wait a few minutes for API rate limits to reset")
    print("=" * 60)
    
    # Keep the tunnel alive
    print("\n✨ App is running. Press Ctrl+C in the notebook to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Gift Genie...")
        ngrok.disconnect(public_url)
        
except Exception as e:
    print(f"\n❌ Error creating tunnel: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check if ngrok authtoken is valid")
    print("2. Verify Streamlit is running: !ps aux | grep streamlit")
    print("3. Check port 8501 is available")
    print("4. Try restarting the notebook kernel")



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

