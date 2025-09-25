import streamlit as st
from advanced_rag import rag_query, setup_vector_database
import os

# Set page config
st.set_page_config(page_title="Real Estate AI Assistant", page_icon="🏠")

# Initialize the database once
if 'db_initialized' not in st.session_state:
    setup_vector_database()
    st.session_state.db_initialized = True

# Title
st.title("🏠 Real Estate AI Assistant")
st.write("Ask me anything about Austin real estate!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about Austin real estate?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_query(prompt)
        st.markdown(response)
    
    # Add AI response to session
    st.session_state.messages.append({"role": "assistant", "content": response})