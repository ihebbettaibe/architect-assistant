"""
Enhanced Streamlit Budget App with LangChain Agent and LangSmith Integration
"""

import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Page configuration
st.set_page_config(
    page_title="🏠 LangChain Budget Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .agent-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .chat-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .langsmith-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 LangChain Budget Agent</h1>
    <p>Assistant immobilier intelligent avec traçage LangSmith</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'langsmith_enabled' not in st.session_state:
    st.session_state.langsmith_enabled = False

@st.cache_resource
def initialize_langchain_agent():
    """Initialize the LangChain budget agent"""
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        
        if not groq_api_key:
            st.error("❌ GROQ_API_KEY not found in environment variables")
            return None, False
            
        # Import and create agent
        from agents.budget.langchain_budget_agent import LangChainBudgetAgent
        
        agent = LangChainBudgetAgent(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            data_folder="cleaned_data",
            use_couchdb=False
        )
        
        langsmith_enabled = bool(langsmith_api_key and agent.tracer)
        
        return agent, langsmith_enabled
        
    except Exception as e:
        st.error(f"❌ Error initializing agent: {e}")
        return None, False

# Sidebar
with st.sidebar:
    st.header("🤖 Agent Status")
    
    # Initialize agent
    if st.session_state.agent is None:
        with st.spinner("🔄 Initializing LangChain agent..."):
            agent, langsmith_enabled = initialize_langchain_agent()
            st.session_state.agent = agent
            st.session_state.langsmith_enabled = langsmith_enabled
    
    # Agent status
    if st.session_state.agent:
        st.success("✅ LangChain Agent Ready")
        
        # LangSmith status
        if st.session_state.langsmith_enabled:
            st.markdown("""
            <div class="langsmith-info">
                🔍 <strong>LangSmith Enabled</strong><br>
                Execution traces available at:<br>
                <a href="https://smith.langchain.com/" target="_blank" style="color: white;">smith.langchain.com</a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("""
            ⚠️ LangSmith Disabled
            
            To enable tracing:
            1. Get API key from smith.langchain.com
            2. Add LANGSMITH_API_KEY to .env file
            3. Restart the app
            """)
    else:
        st.error("❌ Agent initialization failed")
        st.stop()
    
    st.divider()
    
    # Conversation stats
    st.subheader("📊 Session Stats")
    st.metric("Messages", len(st.session_state.conversation_history))
    
    if st.session_state.agent:
        context_summary = st.session_state.agent.get_context_summary()
        st.caption(f"Context: {context_summary}")
    
    # Clear conversation
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        if st.session_state.agent:
            st.session_state.agent.clear_memory()
        st.rerun()

# Main chat interface
st.subheader("💬 Chat with LangChain Agent")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### 📜 Conversation History")
    
    for i, exchange in enumerate(st.session_state.conversation_history):
        with st.expander(f"Exchange {i+1} - {exchange['timestamp']}", expanded=i==len(st.session_state.conversation_history)-1):
            st.markdown(f"**🧑 User:** {exchange['user_message']}")
            st.markdown(f"**🤖 Agent:** {exchange['agent_response']}")
            
            if exchange.get('properties'):
                st.caption(f"Found {len(exchange['properties'])} properties")
            
            if exchange.get('context'):
                st.caption(f"Context: {len(exchange['context'])} items")

# Chat input
st.markdown("### 💭 Ask the Agent")

# Example prompts
examples = [
    "Je cherche une propriété avec un budget de 350 000 DT à Tunis",
    "Quelle est la propriété avec la plus grande surface dans mon budget?",
    "Montre-moi des options moins chères à Sousse",
    "Quel budget prévoir pour une villa de 200 m² à La Marsa?"
]

selected_example = st.selectbox(
    "💡 Or choose an example:",
    [""] + examples,
    index=0
)

user_input = st.text_area(
    "Your message:",
    value=selected_example,
    height=100,
    placeholder="Describe your real estate project, budget, and preferences..."
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)

# Process message
if send_button and user_input.strip():
    if st.session_state.agent:
        # Create callback handler for Streamlit
        from agents.budget.langchain_budget_agent import StreamlitCallbackHandler
        callback_container = st.container()
        callback_handler = StreamlitCallbackHandler(callback_container)
        
        with st.spinner("🤖 Agent is thinking..."):
            try:
                # Get response from agent
                result = st.session_state.agent.chat(user_input, callback_handler)
                
                # Display response
                if result and result.get('response'):
                    st.success("✅ Response received!")
                    
                    # Show response
                    st.markdown("### 🤖 Agent Response")
                    st.markdown(result['response'])
                    
                    # Show properties if found
                    if result.get('properties'):
                        st.markdown(f"### 🏠 Properties Found ({len(result['properties'])})")
                        
                        for i, prop in enumerate(result['properties'][:5], 1):
                            with st.expander(f"Property {i}: {prop.get('Title', 'N/A')[:50]}..."):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**Price:** {prop.get('Price', 0):,} DT")
                                    st.write(f"**Surface:** {prop.get('Surface', 0)} m²")
                                    st.write(f"**Location:** {prop.get('Location', 'N/A')}")
                                    st.write(f"**Type:** {prop.get('Type', 'N/A')}")
                                
                                with col2:
                                    if prop.get('budget_fit_score'):
                                        st.metric("Fit Score", f"{prop['budget_fit_score']:.1f}/10")
                                    
                                    if prop.get('URL'):
                                        st.markdown(f"[View Listing]({prop['URL']})")
                    
                    # Show context
                    if result.get('context'):
                        with st.expander("🔍 Agent Context"):
                            st.json(result['context'])
                    
                    # Add to conversation history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.conversation_history.append({
                        'timestamp': timestamp,
                        'user_message': user_input,
                        'agent_response': result['response'],
                        'properties': result.get('properties', []),
                        'context': result.get('context', {})
                    })
                    
                    # LangSmith info
                    if st.session_state.langsmith_enabled:
                        st.info(f"""
                        🔍 **LangSmith Tracing Active**
                        
                        This conversation has been traced in LangSmith. You can view:
                        - Tool usage and decisions
                        - LLM prompts and responses  
                        - Performance metrics
                        - Error tracking
                        
                        Visit [LangSmith Dashboard](https://smith.langchain.com/) to see detailed traces.
                        """)
                
                else:
                    st.error("❌ No response received from agent")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
                
                # Show error details if available
                if hasattr(e, '__dict__'):
                    with st.expander("🐛 Error Details"):
                        st.json(e.__dict__)
    else:
        st.error("❌ Agent not initialized")

elif send_button:
    st.warning("⚠️ Please enter a message")

# Footer
st.divider()

# LangSmith setup instructions
if not st.session_state.langsmith_enabled:
    with st.expander("🔧 Enable LangSmith Tracing", expanded=False):
        st.markdown("""
        ### 🚀 Enable Advanced Tracing with LangSmith
        
        LangSmith provides detailed insights into your agent's behavior:
        
        **Features:**
        - 🔍 Trace every tool call and decision
        - 📊 Performance monitoring
        - 🐛 Error tracking and debugging
        - 📈 Usage analytics
        
        **Setup Steps:**
        1. **Sign up** at [smith.langchain.com](https://smith.langchain.com/)
        2. **Get API Key** from your settings
        3. **Add to .env file:**
           ```
           LANGSMITH_API_KEY=your_api_key_here
           ```
        4. **Restart** the Streamlit app
        
        **Benefits:**
        - Debug complex agent workflows
        - Monitor performance over time
        - Optimize prompts and tools
        - Track user interactions
        """)

# Technical info
with st.expander("🔧 Technical Details", expanded=False):
    st.markdown(f"""
    ### 🏗️ Architecture
    
    **Agent Type:** LangChain with Groq API
    **Model:** mixtral-8x7b-32768
    **Tools:** {len(st.session_state.agent.tools) if st.session_state.agent else 0} specialized tools
    **Memory:** Conversation buffer (10 exchanges)
    **Tracing:** {'✅ LangSmith' if st.session_state.langsmith_enabled else '❌ Disabled'}
    
    ### 🛠️ Available Tools
    """)
    
    if st.session_state.agent:
        for tool in st.session_state.agent.tools:
            st.write(f"- **{tool.name}:** {tool.description}")

st.markdown("""
---
<div style="text-align: center; padding: 1rem; opacity: 0.7;">
    🏠 LangChain Budget Agent • Enhanced with LangSmith Tracing • Real Estate Intelligence
</div>
""", unsafe_allow_html=True)
