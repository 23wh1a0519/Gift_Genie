# 🎁 Gift Genie - Agentic AI Gifting Assistant

Gift Genie is an Agentic AI app built using **LangChain**, **Gemini Pro**, and **Streamlit** to help you find creative, personalized gifts based on the recipient's interests and budget.

## 🌟 Features

- AI-powered gift suggestions using Gemini 1.5 Pro
- Interest extraction & product matching via LangChain tools
- Simple and clean Streamlit UI
- Shareable using ngrok + Colab (no deployment needed)

## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/23wh1a0519/genie.git
cd genie
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a file named `.env`:

```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 4. Run locally

```bash
streamlit run app.py
```

Or use Colab with `colab_runner.py` to expose using ngrok.

## 📸 Screenshot

![Gift Genie UI](https://via.placeholder.com/800x400?text=Screenshot+Placeholder)

## 🧠 Powered by

- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Ngrok](https://ngrok.com/)