import streamlit as st
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add the agents directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

# Import the simplified LangChain budget agent
try:
    from agents.budget.langchain_budget_agent import LangChainBudgetAgent
    from agents.budget.simple_langchain_agent import SimpleLangChainBudgetAgent
    from agents.budget.minimal_budget_agent import MinimalBudgetAgent
    from agents.budget.simple_fallback_agent import SimpleFallbackBudgetAgent
    from agents.budget.standalone_fallback_agent import StandaloneFallbackBudgetAgent
    import os
    from dotenv import load_dotenv
    load_dotenv()
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ LangChain agent not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="🏠 Architecture Budget Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #10a37f;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(16, 163, 127, 0.1);
    }
    
    /* ChatGPT-like message styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0;
    }
    
    .user-message {
        background: #a2d2ff;
        padding: 20px;
        margin: 10px 0;
        border-radius: 18px;
        border: 1px solid #bae6fd;
        margin-left: 50px;
        position: relative;
    }
    
    .user-message::before {
        content: "🙋‍♂️";
        position: absolute;
        left: -35px;
        top: 15px;
        font-size: 20px;
        background: white;
        border-radius: 50%;
        padding: 5px;
        border: 2px solid #0ea5e9;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        padding: 20px;
        margin: 10px 0;
        border-radius: 18px;
        margin-right: 50px;
        position: relative;
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.15);
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
    
    .agent-message a {
        color: #FFE135;
        text-decoration: none;
        font-weight: bold;
        padding: 2px 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    .agent-message a:hover {
        color: #FFF;
        background: rgba(255, 255, 255, 0.2);
        text-decoration: underline;
    }
    
    /* Property card styling */
    .property-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #10a37f;
    }
    
    .property-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .property-title {
        color: #10a37f;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 12px;
        line-height: 1.3;
    }
    
    .property-price {
        color: #e53e3e;
        font-weight: bold;
        font-size: 22px;
        margin-bottom: 8px;
    }
    
    .property-details {
        color: #4a5568;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    
    .property-url {
        margin-top: 12px;
    }
    
    .property-url a {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 25px;
        text-decoration: none;
        font-size: 14px;
        font-weight: bold;
        transition: all 0.3s ease;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(16, 163, 127, 0.3);
    }
    
    .property-url a:hover {
        background: linear-gradient(135deg, #0d8f6b 0%, #0a7860 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.4);
    }
    
    /* Chat input styling */
    .stTextArea textarea {
        border-radius: 24px;
        border: 2px solid #e2e8f0;
        padding: 16px 20px;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTextArea textarea:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
        outline: none;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        border: none;
        border-radius: 24px;
        padding: 12px 32px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 163, 127, 0.4);
        background: linear-gradient(135deg, #0d8f6b 0%, #0a7860 100%);
    }
    
    /* Example button styling */
    .example-btn {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 20px;
        padding: 12px 20px;
        margin: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        color: #2d3748;
        font-weight: 500;
    }
    
    .example-btn:hover {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        border-color: #10a37f;
        color: #234e52;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metrics container */
    .metrics-container {
        background: linear-gradient(135deg, #10a37f 0%, #0d8f6b 100%);
        color: white;
        padding: 24px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(16, 163, 127, 0.25);
    }
    
    /* Top properties section */
    .top-properties {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 20px;
        padding: 24px;
        margin: 24px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    
    .top-properties h3 {
        color: #1a202c;
        margin-bottom: 20px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #10a37f, #0d8f6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner("🚀 Initializing Budget Agent..."):
        try:
            # Get Groq API key from environment
            groq_api_key = os.getenv('GROQ_API_KEY')
            groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            
            if not groq_api_key:
                st.warning("⚠️ GROQ_API_KEY not found - using standalone fallback agent")
            
            # Set environment variables to avoid PyTorch issues
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Try LangChain agent first if API key is available
            if groq_api_key and LANGCHAIN_AVAILABLE:
                try:
                    print("🔄 Starting LangChain agent initialization...")
                    st.session_state.agent = LangChainBudgetAgent(
                        groq_api_key=groq_api_key,
                        model_name=groq_model,
                        data_folder="cleaned_data",
                        use_couchdb=False
                    )
                    st.session_state.initialized = True
                    st.session_state.agent_type = "langchain"
                    st.success("✅ LangChain agent initialized successfully")
                    print("✅ LangChain agent initialization completed successfully")
                    
                    # Check if LangSmith is enabled
                    if st.session_state.agent.tracer:
                        st.info("🔍 LangSmith tracing is enabled! View traces at https://smith.langchain.com/")
                    
                except Exception as e:
                    st.warning(f"⚠️ LangChain agent failed: {str(e)}")
                    raise e  # Continue to fallback
            else:
                raise Exception("No API key or LangChain not available - using fallback")
            
        except Exception as e:
            # Try fallback to minimal agent
            try:
                st.info("🔄 Trying fallback to minimal agent...")
                st.session_state.agent = MinimalBudgetAgent(
                    data_folder="cleaned_data",
                    use_couchdb=False
                )
                st.session_state.initialized = True
                st.session_state.agent_type = "minimal"
                st.success("✅ Minimal agent initialized successfully (fallback mode)")
                
            except Exception as fallback_error:
                st.warning(f"⚠️ Minimal agent also failed: {fallback_error}")
                
                # Try simple fallback agent
                try:
                    st.info("🔄 Trying simple fallback agent...")
                    st.session_state.agent = SimpleFallbackBudgetAgent(
                        data_folder="cleaned_data"
                    )
                    st.session_state.initialized = True
                    st.session_state.agent_type = "fallback"
                    st.success("✅ Simple fallback agent initialized successfully")
                    
                except Exception as simple_error:
                    st.warning(f"⚠️ Simple fallback agent failed: {simple_error}")
                    
                    # Try standalone fallback agent (final fallback)
                    try:
                        st.info("🔄 Trying standalone fallback agent...")
                        st.session_state.agent = StandaloneFallbackBudgetAgent(
                            data_folder="cleaned_data"
                        )
                        st.session_state.initialized = True
                        st.session_state.agent_type = "standalone_fallback"
                        st.success("✅ Standalone fallback agent initialized successfully")
                        
                    except Exception as final_error:
                        st.error(f"❌ All agents failed. Final error: {final_error}")
                        
                        # Provide specific solutions for common errors
                        error_msg = str(e)
                        if "meta tensor" in error_msg.lower():
                            st.error("🔧 PyTorch meta tensor error detected. Try:")
                            st.code("pip uninstall torch torchvision torchaudio transformers -y")
                            st.code("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
                            st.code("pip install transformers")
                        elif "groq" in error_msg.lower():
                            st.error("🔧 Groq API error. Check your GROQ_API_KEY in .env file")
                        
                        st.session_state.initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Main title
st.markdown('<h1 class="main-header">🏠 Architecture Budget Assistant</h1>', unsafe_allow_html=True)

# Sidebar for agent status and options
with st.sidebar:
    st.header("🤖 Agent Status")
    
    if st.session_state.get('initialized', False):
        agent_type = st.session_state.get('agent_type', 'unknown')
        if agent_type == "langchain":
            st.success("✅ LangChain Budget Agent Ready")
        elif agent_type == "minimal":
            st.success("✅ Minimal Budget Agent Ready (Fallback)")
        elif agent_type == "fallback":
            st.success("✅ Simple Fallback Agent Ready")
        else:
            st.success("✅ Budget Agent Ready")
            
        # Get property count from base agent
        try:
            if hasattr(st.session_state.agent, 'base_agent'):
                prop_count = len(st.session_state.agent.base_agent.property_metadata)
            elif hasattr(st.session_state.agent, 'property_metadata'):
                prop_count = len(st.session_state.agent.property_metadata)
            else:
                prop_count = len(st.session_state.agent.property_data) if hasattr(st.session_state.agent, 'property_data') else 0
            st.info(f"📊 Properties Loaded: {prop_count:,}")
        except:
            st.info("📊 Properties: Ready to analyze")
    else:
        st.error("❌ Agent Not Initialized")
    
    st.header("🔧 Options")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["💬 Chat Mode", "📊 Traditional Analysis", "📈 Market Report"],
        index=0
    )
    
    if st.button("🔄 Clear History", help="Clear conversation history"):
        st.session_state.conversation_history = []
        st.session_state.analysis_results = {}
        st.rerun()

# Chat Mode
if mode == "💬 Chat Mode":
    st.header("💬 Chat with Budget Agent")
    
    # Display conversation history with ChatGPT-like styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f'''
            <div class="user-message">
                <strong>Vous:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="agent-message">
                <strong>Assistant Immobilier:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
            
            # Show top properties if this is a property search response
            if 'propriétés trouvées' in message["content"].lower() and st.session_state.get('top_properties'):
                st.markdown('<div class="top-properties">', unsafe_allow_html=True)
                st.markdown('<h4>🏆 Top 3 Propriétés les Plus Compatibles</h4>', unsafe_allow_html=True)
                
                # Get top properties from session state
                top_properties = st.session_state.get('top_properties', [])
                
                for idx, prop in enumerate(top_properties[:3], 1):
                    compatibility_score = prop.get('compatibility_score', 0.8) * 100
                    property_card = f'''
                    <div class="property-card">
                        <div class="property-title">#{idx} - {prop.get('Title', 'Propriété')} 
                            <span style="color: #10a37f; font-size: 14px;">({compatibility_score:.0f}% compatible)</span>
                        </div>
                        <div class="property-price">{prop.get('Price', 0):,.0f} TND</div>
                        <div class="property-details">
                            📍 <strong>Localisation:</strong> {prop.get('Location', 'N/A')}<br>
                            📐 <strong>Surface:</strong> {prop.get('Surface', 0):.0f} m²<br>
                            🏠 <strong>Type:</strong> {prop.get('Type', 'N/A')}<br>
                            💵 <strong>Prix/m²:</strong> {prop.get('price_per_m2', 0):,.0f} TND/m²<br>
                            💡 <strong>Pourquoi compatible:</strong> {prop.get('why_compatible', 'Propriété recommandée')}
                        </div>
                        <div class="property-url">
                            <a href="{prop.get('URL', '#')}" target="_blank">🔗 Voir les Détails</a>
                        </div>
                    </div>
                    '''
                    st.markdown(property_card, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with improved styling
    user_input = st.text_area(
        "💬 Votre Message:",
        placeholder="Posez-moi des questions sur les budgets, prix immobiliers, analyses de marché...",
        height=100,
        key="chat_input",
        help="Décrivez votre projet immobilier avec votre budget et la ville souhaitée"
    )
    
    # Button layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        send_button = st.button("🚀 Envoyer le Message", type="primary", key="send_msg")
    
    with col2:
        # Show top 3 properties button (only if we have search results)
        if st.session_state.get('top_properties'):
            if st.button("🏆 Top 3 Propriétés", help="Afficher les 3 propriétés les plus compatibles", key="show_top3"):
                st.session_state.show_top_properties = True
                st.rerun()
    
    # Handle send message
    if send_button and user_input.strip():
        if st.session_state.get('initialized', False):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            
            with st.spinner("🤖 L'assistant réfléchit..."):
                try:
                    # Process with different methods based on agent type
                    if st.session_state.agent_type == "standalone_fallback":
                        # Use process_query method for standalone agent
                        result = st.session_state.agent.process_query(user_input)
                        
                        # Format response for standalone agent
                        agent_response = result if isinstance(result, str) else str(result)
                        
                        # Try to get search results for property display
                        try:
                            search_result = st.session_state.agent.search_properties(user_input, limit=10)
                            if search_result.get('properties'):
                                # Convert to format expected by UI
                                properties = search_result['properties']
                                top_properties = properties[:3]  # Get top 3
                                
                                # Convert to expected format
                                formatted_props = []
                                for prop in top_properties:
                                    formatted_prop = {
                                        'Title': f"{prop['category']} - {prop['city']}",
                                        'Price': prop['price'],
                                        'Location': prop['location'],
                                        'Surface': prop['surface'],
                                        'Type': prop['category'],
                                        'price_per_m2': prop['price_per_m2'],
                                        'URL': prop.get('link', '#'),
                                        'compatibility_score': prop.get('score', 0.8),
                                        'why_compatible': f"Surface {prop['surface']:.0f}m², prix {prop['price']:,.0f} TND"
                                    }
                                    formatted_props.append(formatted_prop)
                                
                                st.session_state.top_properties = formatted_props
                                
                                # Add property count info to response
                                total_found = search_result.get('total_found', len(properties))
                                agent_response += f"\n\n📊 **{total_found} propriétés trouvées** correspondant à vos critères."
                                if formatted_props:
                                    agent_response += f"\n🏆 **Top 3 propriétés les plus compatibles disponibles ci-dessous!**"
                        except Exception as e:
                            print(f"Error getting search results for standalone agent: {e}")
                    
                    else:
                        # Use chat method for other agents
                        result = st.session_state.agent.chat(user_input)
                        
                        # Format agent response
                        agent_response = result.get("response", "Je n'ai pas pu traiter votre demande.")
                        
                        # Show error details if debugging
                        if "Je rencontre des difficultés techniques" in agent_response and result.get("error"):
                            st.error(f"Debug - Error details: {result['error']}")
                        
                        # Get top properties if available
                        if result.get("context") and result["context"].get("user_budget") and result["context"].get("preferred_city"):
                            try:
                                # Get detailed analysis with properties
                                client_profile = {
                                    "budget": result["context"]["user_budget"],
                                    "city": result["context"]["preferred_city"],
                                    "preferences": "maison villa terrain",
                                    "min_size": 100,
                                    # Remove max_price limitation to get broader property range
                                    # "max_price": result["context"]["user_budget"]  # REMOVED
                                }
                                
                                # Get analysis from agent (different method for fallback agent)
                                if st.session_state.agent_type == "fallback":
                                    analysis = st.session_state.agent.analyze_client_budget(client_profile)
                                else:
                                    analysis = st.session_state.agent.base_agent.analyze_client_budget(client_profile)
                                
                                if analysis.get('comparable_properties'):
                                    # Store analysis results in agent context
                                    st.session_state.agent.context["analysis_results"] = analysis
                                    
                                    # Get top 3 most compatible properties
                                    top_properties = st.session_state.agent.get_top_compatible_properties(3)
                                    
                                    if top_properties:
                                        st.session_state.top_properties = top_properties
                                        st.session_state.last_search_results = analysis
                                        
                                        # Add property count and URL info to response
                                        properties_count = len(analysis['comparable_properties'])
                                        agent_response += f"\n\n📊 **{properties_count} propriétés trouvées** correspondant à vos critères."
                                        agent_response += f"\n🏆 **Top 3 propriétés les plus compatibles disponibles ci-dessous avec liens directs!**"
                            
                            except Exception as e:
                                print(f"Error getting top properties: {e}")
                        
                        # Add context information
                        if result.get("context"):
                            context = result["context"]
                            if context.get("user_budget"):
                                agent_response += f"\n\n💰 **Budget détecté:** {context['user_budget']:,} TND"
                            if context.get("preferred_city"):
                                agent_response += f"\n📍 **Ville:** {context['preferred_city']}"
                    
                    st.session_state.conversation_history.append({"role": "agent", "content": agent_response})
                    
                except Exception as e:
                    error_msg = f"Désolé, j'ai rencontré une erreur: {str(e)}"
                    st.session_state.conversation_history.append({"role": "agent", "content": error_msg})
            
            st.rerun()
        else:
            st.error("❌ Agent not initialized. Please refresh the page.")
    
    # Display top 3 properties if requested
    if st.session_state.get('show_top_properties', False) and st.session_state.get('top_properties'):
        st.markdown("---")
        st.markdown('<div class="top-properties">', unsafe_allow_html=True)
        st.markdown('<h3>🏆 Top 3 Propriétés les Plus Compatibles</h3>', unsafe_allow_html=True)
        
        top_properties = st.session_state.get('top_properties', [])
        
        for idx, prop in enumerate(top_properties[:3], 1):
            compatibility_score = prop.get('compatibility_score', 0.8) * 100
            property_card = f'''
            <div class="property-card">
                <div class="property-title">#{idx} - {prop.get('Title', 'Propriété')} 
                    <span style="color: #10a37f; font-size: 14px;">({compatibility_score:.0f}% compatible)</span>
                </div>
                <div class="property-price">{prop.get('Price', 0):,.0f} TND</div>
                <div class="property-details">
                    📍 <strong>Localisation:</strong> {prop.get('Location', 'N/A')}<br>
                    📐 <strong>Surface:</strong> {prop.get('Surface', 0):.0f} m²<br>
                    🏠 <strong>Type:</strong> {prop.get('Type', 'N/A')}<br>
                    💵 <strong>Prix/m²:</strong> {prop.get('price_per_m2', 0):,.0f} TND/m²<br>
                    💡 <strong>Pourquoi compatible:</strong> {prop.get('why_compatible', 'Propriété recommandée')}
                </div>
                <div class="property-url">
                    <a href="{prop.get('URL', '#')}" target="_blank">🔗 Voir les Détails</a>
                </div>
            </div>
            '''
            st.markdown(property_card, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Button to hide top properties
        if st.button("❌ Fermer Top 3", key="close_top3"):
            st.session_state.show_top_properties = False
            st.rerun()

# Traditional Analysis Mode
elif mode == "📊 Traditional Analysis":
    st.header("📊 Traditional Budget Analysis")
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox(
                "🏙️ Target City:",
                ["Tunis", "Sousse", "Sfax", "Ariana", "Ben Arous", "Bizerte", "Kairouan", "Mahdia", "Monastir"]
            )
            
            budget = st.number_input(
                "💰 Budget (DT):",
                min_value=50000,
                max_value=5000000,
                value=300000,
                step=10000
            )
            
            preferences = st.text_input(
                "🏠 Property Preferences:",
                value="terrain villa appartement",
                help="Describe what type of property you're looking for"
            )
        
        with col2:
            min_size = st.number_input(
                "📐 Minimum Surface (m²):",
                min_value=50,
                max_value=2000,
                value=150,
                step=10
            )
            
            max_price = st.number_input(
                "💸 Maximum Price (DT):",
                min_value=budget,
                max_value=budget * 2,
                value=int(budget * 1.2),
                step=10000
            )
        
        submitted = st.form_submit_button("🔍 Analyze Market", type="primary")
    
    if submitted:
        client_profile = {
            "city": city,
            "budget": budget,
            "preferences": preferences,
            "min_size": min_size,
            "max_price": max_price
        }
        
        with st.spinner("📊 Analyzing market data..."):
            try:
                # Use the appropriate agent's analysis method
                if st.session_state.agent_type == "fallback":
                    analysis = st.session_state.agent.analyze_client_budget(client_profile)
                else:
                    analysis = st.session_state.agent.base_agent.analyze_client_budget(client_profile)
                st.session_state.analysis_results = analysis
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🏠 Properties Analyzed", 
                        f"{analysis.get('total_properties_analyzed', 0):,}"
                    )
                
                with col2:
                    st.metric(
                        "✅ Matching Properties", 
                        analysis['market_statistics']['inventory_count']
                    )
                
                with col3:
                    if analysis['market_statistics']['inventory_count'] > 0:
                        feasibility = analysis['market_statistics']['budget_feasibility']['feasibility_ratio']
                        st.metric("💯 Budget Feasibility", f"{feasibility:.1%}")
                    else:
                        st.metric("💯 Budget Feasibility", "N/A")
                
                with col4:
                    confidence = analysis['budget_analysis']['confidence_score']
                    st.metric("🎯 Confidence Score", f"{confidence:.1%}")
                
                # Detailed results
                if analysis['market_statistics']['inventory_count'] > 0:
                    market_stats = analysis['market_statistics']
                    budget_ai = analysis['budget_analysis']
                    
                    # Market Statistics
                    st.subheader("📈 Market Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**💰 Price Range:** {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT")
                        st.write(f"**📊 Average Price:** {market_stats['price_stats']['mean']:,.0f} DT")
                        st.write(f"**📊 Median Price:** {market_stats['price_stats']['median']:,.0f} DT")
                    
                    with col2:
                        st.write(f"**📏 Average Surface:** {market_stats['surface_stats']['mean']:.0f} m²")
                        st.write(f"**💵 Avg Price/m²:** {market_stats['price_per_m2_stats']['mean']:,.0f} DT/m²")
                    
                    # AI Recommendations
                    st.subheader("🎯 AI Recommendations")
                    
                    st.write(f"**✅ Budget Validation:** {budget_ai['budget_validation']}")
                    
                    if 'market_position' in budget_ai:
                        st.info(f"**📍 Market Position:** {budget_ai['market_position']}")
                    
                    st.write(f"**💡 Recommendations:** {budget_ai['recommendations']}")
                    
                    # Display the most compatible property with URL
                    if analysis.get('comparable_properties'):
                        if st.session_state.agent_type == "fallback":
                            most_compatible_props = st.session_state.agent.get_top_compatible_properties(1)
                            most_compatible = most_compatible_props[0] if most_compatible_props else None
                        else:
                            most_compatible = st.session_state.agent.base_agent.get_most_compatible_property(client_profile, analysis['comparable_properties'])
                        
                        if most_compatible:
                            st.subheader("🏆 Most Compatible Property")
                            
                            if st.session_state.agent_type == "fallback":
                                # For fallback agent, the property is already in the right format
                                prop = most_compatible
                                compatibility_score = prop.get('compatibility_score', 0.8)
                                why_compatible = prop.get('why_compatible', 'Propriété recommandée')
                            else:
                                # For other agents, extract from the structure
                                prop = most_compatible['property_details']
                                compatibility_score = most_compatible['compatibility_score']
                                why_compatible = most_compatible['why_compatible']
                            
                            # Create two columns for property details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**🏠 Title:** {prop['Title']}")
                                st.write(f"**💰 Price:** {prop['Price']:,.0f} DT")
                                st.write(f"**📐 Surface:** {prop['Surface']:.0f} m²")
                                st.write(f"**💵 Price per m²:** {prop['price_per_m2']:,.0f} DT/m²")
                            
                            with col2:
                                st.write(f"**📍 Location:** {prop['Location']}")
                                st.write(f"**🏗️ Type:** {prop.get('Type', 'N/A')}")
                                st.write(f"**🌟 Compatibility:** {compatibility_score:.1%}")
                            
                            # Display URL as a clickable link
                            if prop.get('URL') and prop['URL'] != 'No URL available':
                                st.markdown(f"**🔗 View Property:** [Click here to view property details]({prop['URL']})")
                            else:
                                st.write("**🔗 URL:** Not available")
                            
                            # Display compatibility explanation
                            st.info(f"**💡 Why this property matches:** {why_compatible}")
                    
                    # Additional insights in expandable sections
                    if 'price_negotiation_tips' in budget_ai:
                        with st.expander("💬 Price Negotiation Tips"):
                            st.write(budget_ai['price_negotiation_tips'])
                    
                    if 'alternative_suggestions' in budget_ai:
                        with st.expander("🔄 Alternative Suggestions"):
                            st.write(budget_ai['alternative_suggestions'])
                    
                    if 'market_trends' in budget_ai:
                        with st.expander("📈 Market Trends"):
                            st.write(budget_ai['market_trends'])
                    
                    if 'risk_assessment' in budget_ai:
                        with st.expander("⚠️ Risk Assessment"):
                            st.write(budget_ai['risk_assessment'])
                
                else:
                    st.warning("⚠️ No properties found matching your criteria. Consider adjusting your parameters.")
                    st.write(f"**💡 Recommendation:** {analysis['budget_analysis']['recommendations']}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Market Report Mode
elif mode == "📈 Market Report":
    st.header("📈 Market Report")
    
    # City selection for market report
    report_city = st.selectbox(
        "🏙️ Select City for Market Report:",
        ["All Cities", "Tunis", "Sousse", "Sfax", "Ariana", "Ben Arous", "Bizerte", "Kairouan", "Mahdia", "Monastir"]
    )
    
    if st.button("📊 Generate Market Report", type="primary"):
        st.info("📈 Market Report feature coming soon! Currently focused on budget analysis.")

# Footer
st.markdown("---")
st.markdown("🏠 **Architecture Budget Assistant** - Powered by Simplified LangChain & Groq AI | 🤖 Simplified LangChain Budget Agent")
