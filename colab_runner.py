from pyngrok import ngrok
import threading, time, os
from dotenv import load_dotenv

def run_streamlit():
    os.system("streamlit run app.py")

# Load API key from .env
load_dotenv()

# Start streamlit in background
threading.Thread(target=run_streamlit).start()
time.sleep(5)

# Expose port 8501
public_url = ngrok.connect(addr="8501", proto="http")
print(f"🔗 Public App URL: {public_url}")