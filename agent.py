from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
import os, re

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY", "dummy_key")
)

catalog = [
    {"name": "Smartwatch X1", "category": "tech", "price": 49, "description": "Smartwatch with fitness tracking"},
    {"name": "LED Book Light", "category": "reading", "price": 15, "description": "Clip-on light for night reading"},
]

def extract_interests(text):
    interests = ["tech", "gaming", "fitness", "reading"]
    return [i for i in interests if i in text.lower()]

def search_products(cat: str, budget: float):
    return "\n".join([
        f"{p['name']} - ${p['price']}: {p['description']}"
        for p in catalog if cat in p["category"] and p["price"] <= budget
    ]) or "No matching gifts."

def safe_parse_budget(text):
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else 9999

def safe_search_wrapper(q: str):
    parts = q.split(",")
    category = parts[0].strip().lower()
    budget_text = parts[1] if len(parts) > 1 else ""
    budget = safe_parse_budget(budget_text)
    return search_products(category, budget)

tools = [
    Tool(name="InterestAnalyzer", func=lambda q: ", ".join(extract_interests(q)), description="Extract user interests"),
    Tool(name="GiftSearcher", func=safe_search_wrapper, description="Find gifts using format: category,budget")
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=20,
    max_execution_time=60
)

def run_agent(query):
    return agent.run(query)