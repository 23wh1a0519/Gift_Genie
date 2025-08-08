import sys
sys.path.append('.')

import streamlit as st
from agent import run_agent

st.set_page_config(page_title="🎁 Gift Genie", layout="centered")
st.title("🎁 Gift Genie - AI Gifting Assistant")
st.markdown("Get creative and personalized gift ideas powered by GenAI!")

user_input = st.text_input("Who are you buying a gift for?", "")

if st.button("Get Ideas"):
    if user_input.strip():
        with st.spinner("Genie is thinking..."):
            result = run_agent(user_input)
            st.success(result)