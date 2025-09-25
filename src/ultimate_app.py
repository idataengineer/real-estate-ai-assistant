import streamlit as st
import os
from dotenv import load_dotenv
from multi_agent_system import CustomerAgent
from enhanced_agent import RealEstateAgentWithMemory
import json
import time

load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Real Estate Assistant",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86C1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize agents in session state
if 'customer_agent' not in st.session_state:
    st.session_state.customer_agent = CustomerAgent()

if 'enhanced_agent' not in st.session_state:
    st.session_state.enhanced_agent = RealEstateAgentWithMemory()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = "Multi-Agent System"

# Header
st.markdown('<h1 class="main-header">🏠 AI Real Estate Assistant</h1>', unsafe_allow_html=True)

# Sidebar - Agent Selection and Info
with st.sidebar:
    st.markdown("## 🤖 AI Agent Selection")
    
    agent_mode = st.selectbox(
        "Choose your AI assistant:",
        ["Multi-Agent System", "Enhanced Agent", "Demo Mode"],
        index=0
    )
    st.session_state.agent_mode = agent_mode
    
    st.markdown("---")
    
    if agent_mode == "Multi-Agent System":
        st.markdown("""
        ### 🔬 Multi-Agent Features
        - **Research Agent**: Market analysis
        - **Financial Agent**: Calculations & budgets  
        - **Customer Agent**: Coordination & chat
        - **Smart Routing**: Auto-selects right agent
        """)
    elif agent_mode == "Enhanced Agent":
        st.markdown("""
        ### 💭 Enhanced Agent Features
        - **Conversation Memory**: Remembers you
        - **Property Search**: RAG-powered
        - **Tool Usage**: Calculators & analysis
        - **Personalization**: Learns preferences
        """)
    else:
        st.markdown("""
        ### 🎯 Demo Mode
        - **Quick Responses**: Fast demonstrations
        - **Key Features**: Showcase capabilities
        - **Interview Ready**: Perfect for presentations
        """)
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    st.markdown(f"**Messages**: {len(st.session_state.messages)}")
    st.markdown(f"**Agent Mode**: {agent_mode}")
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.enhanced_agent = RealEstateAgentWithMemory()
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 💬 Chat with Your AI Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about real estate, mortgages, or property analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response based on selected agent
        with st.chat_message("assistant"):
            with st.spinner("AI agents analyzing your request..."):
                
                if agent_mode == "Multi-Agent System":
                    response = st.session_state.customer_agent.coordinate_response(prompt)
                    st.markdown("🤖 **Multi-Agent Analysis:**")
                    
                elif agent_mode == "Enhanced Agent":
                    response = st.session_state.enhanced_agent.chat(prompt)
                    st.markdown("💭 **Enhanced Agent Response:**")
                    
                else:  # Demo Mode
                    response = f"""**Demo Response for: "{prompt}"**
                    
🔍 **Research Agent Found**: Austin market showing 8% growth, tech-driven demand

💰 **Financial Agent Calculated**: Monthly payment ~$2,247 for $450K home (20% down, 6.5% rate)

🏠 **Customer Agent Recommends**: Based on your query, consider properties in 78704 area - family-friendly with good schools

*This is a demo showcasing multi-agent coordination capabilities.*"""
                
                st.markdown(response)
        
        # Add AI response to session
        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.markdown("## 📈 Quick Tools")
    
    # Mortgage Calculator Widget
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown("### 🏦 Quick Mortgage Calculator")
    
    price = st.number_input("Home Price ($)", value=450000, step=10000)
    down_payment = st.number_input("Down Payment (%)", value=20, step=5)
    interest_rate = st.number_input("Interest Rate (%)", value=6.5, step=0.1)
    
    if st.button("💰 Calculate Payment"):
        loan_amount = price - (price * down_payment / 100)
        monthly_rate = interest_rate / 100 / 12
        num_payments = 30 * 12
        
        if monthly_rate == 0:
            monthly_payment = loan_amount / num_payments
        else:
            monthly_payment = loan_amount * (monthly_rate * (monthly_rate + 1)**num_payments) / ((monthly_rate + 1)**num_payments - 1)
        
        st.success(f"**Monthly Payment: ${monthly_payment:,.2f}**")
        st.info(f"Loan Amount: ${loan_amount:,.2f}")
        st.info(f"Total Interest: ${(monthly_payment * num_payments) - loan_amount:,.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample Questions
    st.markdown("## 💡 Try These Questions")
    
    sample_questions = [
        "What's the Austin real estate market like?",
        "Can I afford a $500K home with $90K income?",
        "Compare 123 Main St vs 456 Oak Ave",
        "What neighborhoods are good for families?",
        "Should I invest in Austin real estate?",
        "Remember my budget is $400K with 2 kids"
    ]
    
    for question in sample_questions:
        if st.button(f"🔸 {question}", key=question):
            # Add question to chat
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Built with Advanced RAG, Multi-Agent AI, and Real Estate Domain Expertise</p>
    <p>Powered by DeepSeek API • Vector Database • LangChain • Streamlit</p>
</div>
""", unsafe_allow_html=True)