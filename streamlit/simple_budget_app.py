import streamlit as st
import os
import sys
from pathlib import Path
import json

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the simple budget agent
try:
    load_dotenv()
    BUDGET_AGENT_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Budget agent not available: {e}")
    st.warning(f"Current sys.path: {sys.path}")

# Configure Streamlit page
st.set_page_config(
    page_title="🏠 Simple Budget Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #10a37f;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 20px;
        margin: 10px 0;
        border-radius: 18px;
        border: 1px solid #bbdefb;
        margin-left: 50px;
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        left: -35px;
        top: 15px;
        font-size: 20px;
        background: white;
        border-radius: 50%;
        padding: 5px;
        border: 2px solid #2196f3;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        padding: 20px;
        margin: 10px 0;
        border-radius: 18px;
        margin-right: 50px;
        position: relative;
    }
    
    .agent-message::before {
        content: "🤖";
        position: absolute;
        right: -35px;
        top: 15px;
        font-size: 20px;
        background: white;
        border-radius: 50%;
        padding: 5px;
        border: 2px solid #10a37f;
    }
    
    .stTextArea textarea {
        border-radius: 24px;
        border: 2px solid #e2e8f0;
        padding: 16px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        border: none;
        border-radius: 24px;
        padding: 12px 32px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #0d8f6b 0%, #0a7860 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner("🚀 Initializing Budget Agent..."):
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
                st.session_state.initialized = False
            else:
                st.session_state.agent = SimpleBudgetAgent()
                st.session_state.initialized = True
                st.session_state.agent_type = "simple"
                st.success("✅ Simple Budget Agent initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main title
st.markdown('<h1 class="main-header">🏠 Simple Budget Assistant</h1>', unsafe_allow_html=True)

# Sidebar for agent status and options
with st.sidebar:
    st.header("🤖 Agent Status")
    
    if st.session_state.get('initialized', False):
        st.success("✅ Budget Agent Ready")
        st.info("💡 This is a simple implementation using Groq API")
    else:
        st.error("❌ Agent Not Initialized")
    
    if st.button("🔄 Clear History"):
        st.session_state.conversation_history = []
        st.rerun()

# Display conversation history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.conversation_history:
    if message['role'] == 'user':
        st.markdown(f'''
        <div class="user-message">
            <strong>You:</strong><br>
            {message["content"]}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="agent-message">
            <strong>Assistant:</strong><br>
            {message["content"]}
        </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.text_area(
    "💬 Your Message:",
    placeholder="Ask me about property budgets, market analysis, or real estate advice...",
    height=100,
    key="chat_input"
)

# Send button
col1, col2 = st.columns([3, 1])

with col1:
    send_button = st.button("🚀 Send Message", type="primary", key="send_msg")

# Handle send message
if send_button and user_input.strip():
    if st.session_state.get('initialized', False):
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 Thinking..."):
            try:
                # Get response from agent
                response = st.session_state.agent.chat(user_input)
                
                # Add agent response to history
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                
                # Rerun to update the UI
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
                st.rerun()
    else:
        st.error("❌ Agent not initialized. Please check the logs.")
