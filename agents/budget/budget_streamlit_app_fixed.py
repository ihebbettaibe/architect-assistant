import sys
import os

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, '.env'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import pandas as pd

# Import local modules with proper path handling
try:
    from budget_agent_base import EnhancedBudgetAgent
    from budget_analysis import BudgetAnalysis
    from client_interface import ClientInterface
except ImportError as e:
    st.error(f"Failed to import budget modules: {e}")
    st.stop()

# Simplified - only standard agent (LangChain removed for stability)
LANGCHAIN_AVAILABLE = False

import json

class FullBudgetAgent(EnhancedBudgetAgent, BudgetAnalysis, ClientInterface):
    pass

# Page configuration
st.set_page_config(
    page_title="Agent Budget Immobilier",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .analysis-section {
        background: #7C7F65;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 2rem;
    }
    
    .chat-agent {
        background: #f1f3f4;
        color: #333;
        margin-right: 2rem;
        border-left: 4px solid #667eea;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    /* Input area enhancement */
    .input-container {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 2rem;  
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Progress bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Main header with gradient background
st.markdown("""
<div class="main-header">
    <h1>🚀 Agent Budget Immobilier</h1>
    <p>Votre assistant intelligent pour l'analyse budgétaire et la recherche de propriétés</p>
</div>
""", unsafe_allow_html=True)

# Define function to render property cards
def render_property_card(prop, index):
    """Render a single property card with enhanced UI"""
    price_per_m2 = prop['Price'] / prop['Surface'] if prop['Surface'] > 0 else 0
    
    # Create a card container
    with st.container():
        st.markdown(f"""
        <div style="
            background: #212529;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px  ;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="color: #e5e5e5; margin: 0; font-size: 1.2em;">
                    🏠 {prop['Title'][:60]}{'...' if len(prop['Title']) > 60 else ''}
                </h4>
                <span style="background: #007bff; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.9em;">
                    #{index}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Property details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💰 Prix",
                f"{prop['Price']:,.0f} DT",
                delta=f"{((prop['Price'] - 200000) / 200000 * 100):+.0f}%" if prop['Price'] > 0 else None
            )
        
        with col2:
            st.metric(
                "📐 Surface", 
                f"{prop['Surface']:.0f} m²",
                delta=f"{prop['Surface'] - 150:.0f} m²" if prop['Surface'] > 150 else None
            )
        
        with col3:
            st.metric(
                "📊 Prix/m²",
                f"{price_per_m2:,.0f} DT/m²",
                delta="Compétitif" if price_per_m2 < 1500 else "Premium"
            )
        
        # Location and URL
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📍 **Localisation:** {prop['Location']}")
        
        with col2:
            if prop.get('URL') and prop['URL'] != 'No URL available':
                st.markdown(f"""
                <a href="{prop['URL']}" target="_blank" style="
                    display: inline-block;
                    background: #28a745;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 25px;
                    text-decoration: none;
                    font-weight: bold;
                    text-align: center;
                    transition: background 0.3s ease;
                ">
                    🔗 Voir l'annonce
                </a>
                """, unsafe_allow_html=True)
            else:
                st.caption("🔗 Lien non disponible")
        
        st.markdown("---")

# Initialize agent with enhanced error handling and progress tracking
@st.cache_resource
def initialize_agent():
    """Initialize the budget agent with caching for better performance"""
    try:
        # Try CouchDB first
        print("🔄 Attempting to initialize agent with CouchDB...")
        return FullBudgetAgent(use_couchdb=True)
        
    except Exception as e:
        st.warning(f"⚠️ CouchDB not available: {e}")
        print(f"CouchDB error: {e}")
        
        # Fallback to CSV files
        try:
            # Try different possible paths for the data folder
            possible_paths = [
                "../../cleaned_data",  # From agents/budget/ to root/cleaned_data
                "cleaned_data",        # If running from root
                os.path.join(root_dir, "cleaned_data")  # Using the root directory we set up
            ]
            
            data_folder = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_folder = path
                    break
            
            if data_folder is None:
                st.warning("⚠️ Data folder not found. Using default configuration.")
                return FullBudgetAgent(use_couchdb=False)
            else:
                return FullBudgetAgent(data_folder=data_folder, use_couchdb=False)
        except Exception as csv_error:
            st.error(f"❌ Erreur lors de l'initialisation: {csv_error}")
            return None

# Initialize session state with enhanced structure
def initialize_session_state():
    """Initialize session state variables with default values"""
    defaults = {
        "agent": None,
        "conversation_history": [],
        "conversation_context": [],  # Changed from {} to [] for list-based context
        # No follow-up questions - users type their own
        "analysis_progress": 0,
        "user_preferences": {
            "auto_execute": True,
            "detailed_analysis": True,
            "show_technical_data": False
        },
        "current_analysis": None,
        "market_data_cache": {},
        "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Initialize agent if not already done
if st.session_state.agent is None:
    with st.spinner("🔄 Initialisation de l'agent budget..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        st.session_state.agent = initialize_agent()
        progress_bar.empty()
        
        if st.session_state.agent:
            st.success("✅ Agent budget initialisé avec succès!")
        else:
            st.error("❌ Impossible d'initialiser l'agent")
            st.stop()

# Enhanced sidebar with better organization
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3>📊 Tableau de Bord</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Configuration Section
    st.subheader("🤖 Configuration Agent")
    
    # Agent type selection - simplified to standard only
    st.info("🤖 **Agent Standard Optimisé** - Analyse rapide et fiable")
    agent_type = "Standard"
    
    # Standard agent is ready to use
    st.session_state.agent_type = "standard"
    
    st.divider()
    
      # Session info
    st.subheader("ℹ️ Statut")
    st.metric("Échanges", len(st.session_state.conversation_history))
    
    # User preferences
    st.subheader("⚙️ Préférences")
    st.session_state.user_preferences["auto_execute"] = st.checkbox(
        "Exécution automatique", 
        value=st.session_state.user_preferences["auto_execute"],
        help="Exécuter automatiquement les recherches de propriétés"
    )
    
    st.divider()
    
    # Conversation history with enhanced display
    st.subheader("💬 Historique")
    
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.conversation_context = [
            {
                "role": "system",
                "content": "Vous êtes un assistant immobilier spécialisé dans l'analyse de budget et la recherche de propriétés en Tunisie. Vous vous concentrez particulièrement sur les terrains."
            }
        ]
        # No follow-up questions to reset
        st.session_state.current_analysis = None
        st.rerun()
    
    # Display conversation history with enhanced UI
    if st.session_state.conversation_history:
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
            with st.expander(f"💬 Échange {len(st.session_state.conversation_history) - i}", expanded=False):
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>Client:</strong> {entry['input'][:80]}{'...' if len(entry['input']) > 80 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Handle different entry types (standard vs hybrid)
                if 'result' in entry:
                    # Standard agent entry
                    budget = entry['result']['budget_analysis'].get('extracted_budget', 'Non spécifié')
                    confidence = entry['result'].get('confidence_level', 'low')
                    agent_response = f"Budget: {budget} - Confiance: {confidence}"
                elif 'response' in entry:
                    # Hybrid or LangChain agent entry
                    properties_count = entry.get('properties_count', 0)
                    agent_type = entry.get('agent_type', 'langchain')
                    if agent_type == 'hybrid':
                        agent_response = f"🚀 Agent Hybride - {properties_count} propriétés analysées"
                    else:
                        agent_response = f"Propriétés trouvées: {properties_count}"
                    confidence = 'medium'  # Default for LangChain entries
                else:
                    # Fallback for unknown entry types
                    agent_response = "Réponse disponible"
                    confidence = 'low'
                
                status_class = {
                    'high': 'status-success',
                    'medium': 'status-warning',
                    'low': 'status-error'
                }.get(confidence, 'status-error')
                
                st.markdown(f"""
                <div class="chat-message chat-agent">
                    <span class="status-indicator {status_class}"></span>
                    <strong>Agent:</strong> {agent_response}
                </div>
                """, unsafe_allow_html=True)

def should_trigger_property_search(user_message: str) -> bool:
    """Check if user message contains keywords that should trigger property search"""
    search_keywords = [
        'cherche', 'recherche', 'trouve', 'montre', 'voir', 'propriété', 'terrain', 
        'maison', 'villa', 'appartement', 'disponible', 'acheter', 'achat',
        'options', 'choix', 'proposer', 'suggère', 'recommande'
    ]
    
    user_lower = user_message.lower()
    return any(keyword in user_lower for keyword in search_keywords)

def process_chat_message(user_message: str) -> tuple:
    """Process user message and return formatted response with search trigger flag"""
    try:
        # Store user message in hidden conversation context
        st.session_state.conversation_context.append({
            "role": "user",
            "content": user_message
        })
        
        # Use the standard agent to process the message
        result = st.session_state.agent.process_client_input(user_message)
        
        # Check if we should auto-trigger property search
        should_search = should_trigger_property_search(user_message)
        
        # Format the response in a conversational way
        response_parts = []
        
        # Add budget analysis if available
        if result.get('budget_analysis'):
            budget_info = result['budget_analysis']
            if budget_info.get('extracted_budget'):
                response_parts.append(f"💰 **Budget détecté:** {budget_info['extracted_budget']:,} DT")
            
            if budget_info.get('budget_flexibility'):
                response_parts.append(f"📋 **Flexibilité:** {budget_info['budget_flexibility']}")
        
        # Add recommendations if available
        if result.get('suggestions'):
            response_parts.append("💡 **Mes recommandations:**")
            for suggestion in result['suggestions'][:3]:  # Show top 3
                response_parts.append(f"• {suggestion}")
        
        # No follow-up questions generation - users will type their own questions
        
        # Add reliability info
        if result.get('reliability_score') is not None:
            confidence = result['reliability_score']
            if confidence >= 0.8:
                response_parts.append("✅ **Confiance:** Élevée - Informations suffisantes")
            elif confidence >= 0.6:
                response_parts.append("⚠️ **Confiance:** Moyenne - Plus d'infos seraient utiles")
            else:
                response_parts.append("❗ **Confiance:** Faible - J'ai besoin de plus de détails")
        
        # If we should search and have enough info, add search message
        if (should_search and 
            result.get('budget_analysis', {}).get('extracted_budget') and 
            result.get('reliability_score', 0) >= 0.5):
            response_parts.append("🔍 **Je recherche des propriétés pour vous...**")
        
        # If no specific analysis, provide general response
        if not response_parts:
            response_parts.append("J'ai bien reçu votre message. Pouvez-vous me donner plus de détails sur votre projet immobilier ? Par exemple, votre budget approximatif, la ville qui vous intéresse, ou le type de propriété recherché ?")
        
        # Format response and store in context
        formatted_response = "\n\n".join(response_parts)
        st.session_state.conversation_context.append({
            "role": "assistant",
            "content": formatted_response
        })
        
        return formatted_response, result, should_search
        
    except Exception as e:
        error_msg = f"❌ Je rencontre quelques difficultés techniques. Pouvez-vous reformuler votre question ? Cela m'aiderait beaucoup ! 😊 {str(e)}"
        st.session_state.conversation_context.append({
            "role": "assistant",
            "content": error_msg
        })
        return error_msg, None, False

# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis votre assistant immobilier. Je peux vous aider à analyser votre budget, rechercher des propriétés et vous donner des conseils personnalisés. Comment puis-je vous aider aujourd'hui ?",
        "timestamp": datetime.now().isoformat()
    })
    
    # Initialize hidden conversation context with system message
    if not st.session_state.conversation_context:
        st.session_state.conversation_context = [
            {
                "role": "system",
                "content": "Vous êtes un assistant immobilier spécialisé dans l'analyse de budget et la recherche de propriétés en Tunisie. Vous vous concentrez particulièrement sur les terrains."
            },
            {
                "role": "assistant",
                "content": "👋 Bonjour ! Je suis votre assistant immobilier. Je peux vous aider à analyser votre budget, rechercher des propriétés et vous donner des conseils personnalisés."
            }
        ]

# Chat-style conversation display
st.markdown("### 💬 Conversation")

# Display chat messages with enhanced property rendering
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        # User message (right-aligned, blue)
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 10px 15px; border-radius: 18px 18px 5px 18px; 
                        max-width: 70%; margin-left: 30%;">
                <strong>Vous:</strong><br>
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message (left-aligned, gray)
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background: #f0f2f6; color: #262730; padding: 10px 15px; 
                        border-radius: 18px 18px 18px 5px; max-width: 70%; margin-right: 30%;">
                <strong>🤖 Assistant Budget:</strong><br>
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # If message contains properties, display them with enhanced UI
        if message.get('properties'):
            st.markdown("### 🏠 Propriétés Sélectionnées")
            
            for i, prop in enumerate(message['properties'], 1):
                render_property_card(prop, i)
        
        # No pre-generated follow-up questions - let user type their own questions
                        

            
            st.markdown("---")

# Chat input at the bottom
st.markdown("---")
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Tapez votre message...",
        placeholder="Ex: J'ai un budget de 300000 DT pour acheter un terrain à Sousse",
        key="chat_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤", help="Envoyer le message", use_container_width=True)

# Handle message sending with Enter key or button
if (send_button or st.session_state.get("submit_triggered", False)) and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process with agent
    with st.spinner("🤖 L'assistant réfléchit..."):
        try:
            # Get comprehensive response using budget agent
            assistant_response, result, should_auto_search = process_chat_message(user_input)
            
            # Add assistant response with follow-up flag if questions available
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "has_follow_up": False
            })
            
            # Auto-trigger property search if keywords detected and budget available
            if (should_auto_search and result and 
                result.get('budget_analysis', {}).get('extracted_budget') and
                result.get('reliability_score', 0) >= 0.5):
                
                # Extract budget and city
                budget = result['budget_analysis']['extracted_budget']
                extracted_city = None
                
                # Extract city from user input
                user_lower = user_input.lower()
                tunisia_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous']
                for city in tunisia_cities:
                    if city in user_lower:
                        extracted_city = city.title()
                        break
                
                # Create client profile for property search
                client_profile = {
                    "city": extracted_city or "Sousse",
                    "budget": budget,
                    "preferences": "terrain",  # Always terrain as requested
                    "min_size": 100,
                    "max_price": budget
                }
                
                # Run property analysis
                property_analysis = st.session_state.agent.analyze_client_budget(client_profile)
                
                if property_analysis and property_analysis['market_statistics']['inventory_count'] > 0:
                    properties = property_analysis['comparable_properties'][:5]  # Top 5
                    market_stats = property_analysis['market_statistics']
                    
                    # Create enhanced property search response
                    property_response = f"""🏠 **Propriétés trouvées pour votre budget de {budget:,} DT:**

📊 **Résumé du marché ({extracted_city or 'Région sélectionnée'}):**
• **{market_stats['inventory_count']} propriétés** disponibles
• **Prix moyen:** {market_stats['price_stats']['mean']:,.0f} DT
• **Faisabilité:** {market_stats['budget_feasibility']['feasibility_ratio']:.1%}"""
                    
                    # Add property search result to conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": property_response,
                        "timestamp": datetime.now().isoformat(),
                        "properties": properties,
                        "market_stats": market_stats,
                        "has_follow_up": False
                    })
                    
                    # Update hidden conversation context
                    st.session_state.conversation_context.append({
                        "role": "assistant",
                        "content": property_response + " [Propriétés affichées à l'utilisateur]"
                    })
                else:
                    # No properties found
                    no_props_response = f"""😔 **Aucune propriété trouvée pour votre budget de {budget:,} DT dans {extracted_city or 'la région sélectionnée'}.**

💡 **Suggestions:**
• Élargir la zone de recherche
• Augmenter légèrement le budget
• Chercher dans les zones périphériques  
• Considérer des propriétés à rénover"""
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": no_props_response,
                        "timestamp": datetime.now().isoformat(),
                        "has_follow_up": False
                    })
                    
                    # Update hidden conversation context
                    st.session_state.conversation_context.append({
                        "role": "assistant",
                        "content": no_props_response
                    })
            
        except Exception as e:
            error_response = f"❌ Je rencontre quelques difficultés techniques. Pouvez-vous reformuler votre question ? Erreur: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat(),
                "has_follow_up": False
            })
            
            # Update hidden conversation context
            st.session_state.conversation_context.append({
                "role": "assistant",
                "content": error_response
            })
    
    # Rerun to show the new message
    st.rerun()

# Property search trigger
if len(st.session_state.messages) >= 2:  # At least one exchange
    last_message = st.session_state.messages[-1]
    if (last_message["role"] == "assistant" and 
        "recherche des propriétés" in last_message["content"].lower() and
        last_message.get("result")):
        
        if st.button("🏠 Oui, rechercher des propriétés maintenant", type="primary", use_container_width=True):
            with st.spinner("🔍 Recherche de propriétés en cours..."):
                try:
                    # Get budget info from last result
                    result = last_message["result"]
                    budget = result['budget_analysis'].get('extracted_budget')
                    
                    if budget:
                        # Extract city from conversation
                        user_messages = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'user']
                        extracted_city = None
                        tunisia_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous']
                        
                        for user_msg in user_messages:
                            user_msg_lower = user_msg.lower()
                            for city in tunisia_cities:
                                if city in user_msg_lower:
                                    extracted_city = city.title()
                                    break
                            if extracted_city:
                                break
                        
                        # Create client profile for property search
                        client_profile = {
                            "city": extracted_city or "Sousse",
                            "budget": budget,
                            "preferences": "terrain",  # Always terrain as requested
                            "min_size": 100,
                            "max_price": budget
                        }
                        
                        # Run property analysis
                        property_analysis = st.session_state.agent.analyze_client_budget(client_profile)
                        
                        if property_analysis and property_analysis['market_statistics']['inventory_count'] > 0:
                            properties = property_analysis['comparable_properties'][:5]  # Top 5
                            market_stats = property_analysis['market_statistics']
                            
                            # Create property search response
                            property_response = f"""🏠 **Propriétés trouvées pour votre budget de {budget:,} DT:**

📊 **Résumé du marché:**
• {market_stats['inventory_count']} propriétés disponibles
• Prix moyen: {market_stats['price_stats']['mean']:,.0f} DT
• Faisabilité: {market_stats['budget_feasibility']['feasibility_ratio']:.1%}

🏠 **Top propriétés sélectionnées:**"""
                            
                            # Add property search result to conversation
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": property_response,
                                "timestamp": datetime.now().isoformat(),
                                "properties": properties,
                                "has_follow_up": False
                            })
                            
                            # Update hidden conversation context
                            st.session_state.conversation_context.append({
                                "role": "assistant",
                                "content": property_response + " [Propriétés affichées à l'utilisateur]"
                            })
                            
                            st.rerun()
                        else:
                            # No properties found
                            no_props_response = f"""😔 **Aucune propriété trouvée pour votre budget de {budget:,} DT dans {extracted_city or 'la région sélectionnée'}.**

💡 **Suggestions:**
• Élargir la zone de recherche
• Augmenter légèrement le budget
• Chercher dans les zones périphériques
• Considérer des propriétés à rénover

Voulez-vous que j'ajuste les critères de recherche ?"""
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": no_props_response,
                                "timestamp": datetime.now().isoformat(),
                                "has_follow_up": False
                            })
                            
                            # Update hidden conversation context
                            st.session_state.conversation_context.append({
                                "role": "assistant",
                                "content": no_props_response
                            })
                            st.rerun()
                            
                except Exception as e:
                    error_response = f"❌ Erreur lors de la recherche de propriétés: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_response, 
                        "timestamp": datetime.now().isoformat(),
                        "has_follow_up": False
                    })
                    
                    # Update hidden conversation context
                    st.session_state.conversation_context.append({
                        "role": "assistant",
                        "content": error_response
                    })
                    st.rerun()

# No quick action buttons - we focus on personalized follow-up questions instead

# Clear conversation button
if st.button("🗑️ Nouvelle Conversation", help="Effacer la conversation actuelle"):
    # Keep welcome message but reset everything else
    welcome_msg = st.session_state.messages[0]
    st.session_state.messages = [welcome_msg]
    
    # Reset hidden conversation context
    st.session_state.conversation_context = [
        {
            "role": "system",
            "content": "Vous êtes un assistant immobilier spécialisé dans l'analyse de budget et la recherche de propriétés en Tunisie. Vous vous concentrez particulièrement sur les terrains."
        },
        {
            "role": "assistant",
            "content": "👋 Bonjour ! Je suis votre assistant immobilier. Je peux vous aider à analyser votre budget, rechercher des propriétés et vous donner des conseils personnalisés."
        }
    ]
    
    # No follow-up questions to reset
    
    st.rerun()

# Enhanced footer with additional features
st.markdown("---")
# Help and tips section
with st.expander("💡 Guide d'Utilisation - Assistant Budget", expanded=False):
    st.markdown("""
    #### 🤖 Comment utiliser l'assistant:
    
    **💰 Pour une analyse budgétaire optimale:**
    - Indiquez un montant précis ou une fourchette
    - Précisez la ville qui vous intéresse
    - Mentionnez vos capacités de financement
    - L'assistant extraira automatiquement les informations clés
    
    **🏠 Concernant votre projet:**
    - Type de bien souhaité (nous nous concentrons sur les terrains)
    - Surface approximative souhaitée
    - Zone géographique préférée
    - L'assistant analysera et trouvera les meilleures options
    
    **🎯 Exemples de questions:**
    - "J'ai 250 000 DT pour un terrain à Sousse"
    - "Quelles sont mes options avec un budget de 180 000 DT?"
    - "Je cherche un terrain d'au moins 200m² à Tunis"
    - "Analyse les tendances du marché à Sfax"
    """)

# Footer with branding and info
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #2f6690; border-radius: 15px; margin-top: 2rem;">
    <h4>🏗️ Assistant Budget Immobilier</h4>
    <p style="margin: 0; opacity: 0.8;">
        Votre assistant intelligent pour des projets immobiliers réussis • 
        Session: {session_id} • 
        Version: 2.0 ChatGPT Style
    </p>
</div>
""".format(session_id=st.session_state.conversation_id[-8:]), unsafe_allow_html=True)
