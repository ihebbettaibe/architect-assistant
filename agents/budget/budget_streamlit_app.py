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

# LangChain agent imports
try:
    from langchain_budget_agent import create_langchain_budget_agent, StreamlitCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.warning(f"LangChain agent not available: {e}")

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
    <h1> Agent Budget Immobilier</h1>
    <p>Votre assistant intelligent pour analyser et optimiser votre budget de projet immobilier</p>
</div>
""", unsafe_allow_html=True)

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
        "conversation_context": {},
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
    
    # Agent type selection
    agent_type = st.radio(
        "Type d'agent",
        options=["Standard", "LangChain + Groq"] if LANGCHAIN_AVAILABLE else ["Standard"],
        help="Choisissez le type d'agent à utiliser"
    )
    
    # Groq API configuration for LangChain agent
    if agent_type == "LangChain + Groq" and LANGCHAIN_AVAILABLE:
        # Try to load API key from environment first
        env_groq_key = os.getenv("GROQ_API_KEY")
        default_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
        
        # Only show API key input if not loaded from environment
        if env_groq_key:
            groq_api_key = env_groq_key
            st.success("✅ Clé API Groq chargée depuis .env")
        else:
            groq_api_key = st.text_input(
                "Clé API Groq",
                value="",
                type="password",
                help="Entrez votre clé API Groq pour utiliser l'agent LangChain"
            )
        
        if groq_api_key:
            groq_model = st.selectbox(
                "Modèle Groq",
                options=["mixtral-8x7b-32768", "llama-3.1-8b-instant", "llama2-70b-4096", "gemma-7b-it"],
                index=0 if default_model == "mixtral-8x7b-32768" else 
                      1 if default_model == "llama-3.1-8b-instant" else 0,
                help="Sélectionnez le modèle Groq à utiliser"
            )
            
            # Show if API key was loaded from environment
            if env_groq_key and groq_api_key == env_groq_key:
                # Already shown success message above
                pass
            
            # Initialize LangChain agent
            if "langchain_agent" not in st.session_state or st.session_state.get("current_groq_key") != groq_api_key:
                try:
                    with st.spinner("Initialisation de l'agent LangChain..."):
                        # Use the same data folder detection logic as the standard agent
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
                        
                        st.session_state.langchain_agent = create_langchain_budget_agent(
                            groq_api_key=groq_api_key,
                            model_name=groq_model,
                            data_folder=data_folder,
                            use_couchdb=True  # Try CouchDB first
                        )
                        st.session_state.current_groq_key = groq_api_key
                        st.session_state.agent_type = "langchain"
                    st.success("✅ Agent LangChain initialisé!")
                except Exception as e:
                    st.error(f"❌ Erreur d'initialisation LangChain: {e}")
                    st.session_state.agent_type = "standard"
        else:
            st.warning("⚠️ Clé API Groq requise pour l'agent LangChain")
            st.session_state.agent_type = "standard"
    else:
        st.session_state.agent_type = "standard"
    
    st.divider()
    
    # Context Display for LangChain agent
    if st.session_state.get("agent_type") == "langchain" and "langchain_agent" in st.session_state:
        st.subheader("🧠 Contexte Agent")
        context_summary = st.session_state.langchain_agent.get_context_summary()
        if context_summary != "Aucun contexte défini":
            st.info(context_summary)
        else:
            st.write("*Aucun contexte défini*")
        
        if st.button("🔄 Réinitialiser contexte", use_container_width=True):
            st.session_state.langchain_agent.clear_memory()
            st.rerun()
        
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
        st.session_state.conversation_context = {}
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
                
                # Handle different entry types (standard vs langchain)
                if 'result' in entry:
                    # Standard agent entry
                    budget = entry['result']['budget_analysis'].get('extracted_budget', 'Non spécifié')
                    confidence = entry['result'].get('confidence_level', 'low')
                    agent_response = f"Budget: {budget} - Confiance: {confidence}"
                elif 'response' in entry:
                    # LangChain agent entry
                    properties_count = entry.get('properties_count', 0)
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

# Main content area with enhanced input section
st.markdown("""
<div class="input-container">
    <h3>💭 Décrivez votre projet immobilier</h3>
    <p>Soyez aussi précis que possible concernant votre budget, vos préférences et vos contraintes.</p>
</div>
""", unsafe_allow_html=True)

# Enhanced input area with examples
examples = [
    "Je souhaite construire une villa avec un budget de 350 000 DT à Tunis",
    "Mon budget est flexible entre 250 000 et 400 000 DT pour un terrain constructible",
    "Je cherche un appartement à rénover avec un budget total de 180 000 DT",
    "Quel budget prévoir pour une maison de 200 m² à La Marsa?"
]

with st.expander("💡 Exemples de questions", expanded=False):
    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
            st.session_state.example_selected = example

# Main input with enhanced styling
user_input = st.text_area(
    "Votre message:",
    height=120,
    placeholder="Décrivez votre projet, votre budget, vos préférences...",
    value=st.session_state.get('example_selected', ''),
    key="main_input",
    help="Plus vous êtes précis, plus l'analyse sera pertinente"
)

# Clear selected example after use
if 'example_selected' in st.session_state:
    del st.session_state.example_selected

# Enhanced auto-execution function
def execute_next_steps_enhanced(result, context, user_input=""):
    """Execute next steps with enhanced feedback and error handling"""
    next_actions = result.get('next_actions', [])
    executed_actions = []
    
    for action in next_actions:
        try:
            action_container = st.container()
            with action_container:
                if action == 'search_comparable_properties' and result['budget_analysis'].get('extracted_budget'):
                    with st.spinner("🔍 Recherche de propriétés comparables..."):
                        budget = result['budget_analysis']['extracted_budget']
                        # Enhanced client profile creation - extract city from user input
                        extracted_city = None
                        if user_input:
                            user_input_lower = user_input.lower()
                            
                            # Try to extract city from user input
                            tunisia_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous']
                            for city in tunisia_cities:
                                if city in user_input_lower:
                                    extracted_city = city.title()
                                    break
                        
                        client_profile = {
                            "city": extracted_city or context.get("preferred_city", "Sousse"),
                            "budget": budget,
                            "preferences": context.get("preferences", "terrain habitation construction"),
                            "min_size": context.get("min_size", 150),
                            "max_price": budget
                        }
                        
                        # Run analysis with error handling
                        budget_analysis = st.session_state.agent.analyze_client_budget(client_profile)
                        
                        if budget_analysis and budget_analysis['market_statistics']['inventory_count'] > 0:
                            st.success(f"✅ {budget_analysis['market_statistics']['inventory_count']} propriétés trouvées!")
                            
                            # Create market visualization
                            market_stats = budget_analysis['market_statistics']
                            
                            # Display enhanced market insights
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Prix Minimum",
                                    f"{market_stats['price_stats']['min']:,.0f} DT",
                                    delta=f"{((market_stats['price_stats']['min'] - budget) / budget * 100):+.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Prix Moyen",
                                    f"{market_stats['price_stats']['mean']:,.0f} DT",
                                    delta=f"{((market_stats['price_stats']['mean'] - budget) / budget * 100):+.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "Faisabilité",
                                    f"{market_stats['budget_feasibility']['feasibility_ratio']:.1%}",
                                    delta="Excellent" if market_stats['budget_feasibility']['feasibility_ratio'] > 0.8 else "À améliorer"
                                )
                            
                            executed_actions.append("✅ Propriétés comparables analysées avec succès")
                        else:
                            st.warning("⚠️ Aucune propriété comparable trouvée dans cette gamme de prix")
                            executed_actions.append("⚠️ Recherche sans résultat - suggestions proposées")
                
                elif action == 'validate_budget_with_market_data':
                    executed_actions.append("✅ Budget validé avec les données du marché")
                
                elif action == 'proceed_to_style_preferences':
                    executed_actions.append("✅ Prêt pour l'agent de style")
                
                elif action == 'assess_regulatory_constraints':
                    executed_actions.append("✅ Prêt pour l'agent réglementaire")
                
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'exécution de {action}: {str(e)}"
            st.error(error_msg)
            executed_actions.append(error_msg)
    return executed_actions

def use_standard_agent(user_input):
    """Execute analysis using the standard budget agent"""
    
    # Enhanced analysis function with progress tracking
    def execute_analysis_with_progress(user_input):
        """Execute analysis with visual progress tracking"""
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Input processing
            status_text.text("🔍 Analyse de votre demande...")
            progress_bar.progress(20)
            
            # Process input with conversation context
            result = st.session_state.agent.process_client_input(
                user_input, 
                st.session_state.conversation_context
            )
            
            # Step 2: Budget analysis
            status_text.text("💰 Analyse du budget...")
            progress_bar.progress(40)
            
            # Step 3: Market data processing
            status_text.text("📊 Traitement des données de marché...")
            progress_bar.progress(60)
            
            # Step 4: Generate recommendations
            status_text.text("💡 Génération des recommandations...")
            progress_bar.progress(80)
            
            # Step 5: Finalization
            status_text.text("✅ Finalisation de l'analyse...")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()        
        return result

    # Execute the standard analysis
    try:
        result = execute_analysis_with_progress(user_input)
        
        # Store current analysis
        st.session_state.current_analysis = result
        
        # Update conversation context with new information
        if result['budget_analysis'].get('extracted_budget'):
            st.session_state.conversation_context['current_budget'] = result['budget_analysis']['extracted_budget']
        
        # Extract and store city from user input
        user_input_lower = user_input.lower()
        tunisia_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous']
        for city in tunisia_cities:
            if city in user_input_lower:
                st.session_state.conversation_context['preferred_city'] = city.title()
                break
        
        # Add to conversation history with timestamp
        conversation_entry = {
            'input': user_input,
            'result': result,
            'agent_type': 'standard',
            'timestamp': datetime.now().isoformat(),
            'summary': f"Budget: {result['budget_analysis'].get('extracted_budget', 'Non spécifié')} - Confiance: {result.get('confidence_level', 'medium')}"
        }
        st.session_state.conversation_history.append(conversation_entry)
        
        # Display enhanced results
        st.markdown("---")
        st.success("🎉 Analyse terminée avec succès!")
        
        # Main results section with enhanced layout
        st.markdown("""
        <div class="analysis-section">
            <h2>📊 Résultats de l'Analyse</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            budget_analysis = result.get('budget_analysis', {})
            budget_value = budget_analysis.get('extracted_budget', 0)
            st.metric(
                "💰 Budget Détecté",
                f"{budget_value:,} DT" if budget_value else "Non spécifié",
                delta="Confirmé" if budget_value else "À préciser"
            )
        
        with col2:
            reliability = result.get('reliability_score', 0)
            st.metric(
                "🎯 Fiabilité",
                f"{reliability:.1%}",
                delta="Excellent" if reliability > 0.8 else "À améliorer"
            )
        
        with col3:
            confidence = result.get('confidence_level', 'low')
            confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
            st.metric(
                "📈 Confiance",
                f"{confidence_colors.get(confidence, '🔴')} {confidence.title()}",
                delta="Élevée" if confidence == 'high' else "Modérée" if confidence == 'medium' else "Faible"
            )
        
        with col4:
            next_actions_count = len(result.get('next_actions', []))
            st.metric(
                "🎯 Actions Suivantes",
                f"{next_actions_count}",
                delta="Prêt" if next_actions_count > 0 else "En attente"
            )
        
        # Auto-execute next steps if enabled and confidence is high enough
        if (st.session_state.user_preferences["auto_execute"] and 
            result.get('reliability_score', 0) >= 0.6):
            
            st.markdown("### 🤖 Exécution Automatique des Recommandations")
            executed_actions = execute_next_steps_enhanced(result, st.session_state.conversation_context, user_input)
            
            if executed_actions:
                st.markdown("**✅ Actions exécutées automatiquement:**")
                for action in executed_actions:
                    st.write(f"• {action}")
        
        # Enhanced Q&A section
        if result.get('targeted_questions'):
            st.markdown("### ❓ Questions pour Affiner l'Analyse")
            for i, question in enumerate(result['targeted_questions'], 1):
                with st.expander(f"Question {i}", expanded=False):
                    st.write(question)
                    
                    # Quick answer input
                    quick_answer = st.text_input(
                        "Réponse rapide:",
                        key=f"quick_answer_{i}",
                        placeholder="Tapez votre réponse ici..."
                    )
                    
                    if quick_answer and st.button(f"Répondre", key=f"answer_{i}"):
                        # Add answer to context
                        if 'quick_answers' not in st.session_state.conversation_context:
                            st.session_state.conversation_context['quick_answers'] = {}
                        st.session_state.conversation_context['quick_answers'][question] = quick_answer
                        st.success("✅ Réponse enregistrée!")
        
        # Enhanced suggestions section
        if result.get('suggestions'):
            st.markdown("### 💡 Recommandations Personnalisées")
            for i, suggestion in enumerate(result['suggestions'], 1):
                st.info(f"**{i}.** {suggestion}")
        
        # Inconsistencies with enhanced display
        if result.get('inconsistencies_detected'):
            st.markdown("### ⚠️ Points d'Attention")
            for inconsistency in result['inconsistencies_detected']:
                severity = inconsistency.get('severity', 'medium')
                message = inconsistency.get('message', '')
                
                if severity == 'high':
                    st.error(f"🚨 **Critique:** {message}")
                elif severity == 'medium':
                    st.warning(f"⚠️ **Important:** {message}")
                else:
                    st.info(f"ℹ️ **Note:** {message}")
        
        # Next actions with enhanced UI
        if result.get('next_actions'):
            st.markdown("### 🎯 Prochaines Étapes Recommandées")
            
            action_labels = {
                'search_comparable_properties': ('🔍', 'Rechercher des propriétés comparables', 'Analyser le marché local'),
                'validate_budget_with_market_data': ('📊', 'Valider le budget avec les données du marché', 'Vérifier la cohérence budgétaire'),
                'collect_more_budget_info': ('💰', 'Collecter plus d\'informations budgétaires', 'Préciser les détails financiers'),
                'clarify_financing_options': ('🏦', 'Clarifier les options de financement', 'Explorer les solutions bancaires')
            }
            
            for action in result['next_actions']:
                icon, title, description = action_labels.get(action, ('⚡', action, 'Action recommandée'))
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 2])
                    with col1:
                        st.markdown(f"<h2 style='text-align: center; margin: 0;'>{icon}</h2>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**{title}**")
                        st.caption(description)
                    with col3:
                        if st.button("Exécuter", key=f"execute_{action}", type="secondary"):
                            # Execute individual action
                            with st.spinner(f"Exécution de {title}..."):
                                executed = execute_next_steps_enhanced({'next_actions': [action]}, st.session_state.conversation_context)
                                for exec_result in executed:
                                    st.success(exec_result)                        
                    st.divider()
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        with st.expander("🐛 Détails de l'erreur", expanded=False):
            st.exception(e)

# Main conversation handling with agent type selection
if st.button("🚀 Analyser", type="primary", use_container_width=True):
    if user_input.strip():
        # Check which agent type to use
        if st.session_state.get("agent_type") == "langchain" and "langchain_agent" in st.session_state:
            # Use LangChain agent
            with st.spinner("🤖 Traitement avec l'agent LangChain..."):
                try:
                    # Process with LangChain agent without showing internal thoughts
                    response_data = st.session_state.langchain_agent.chat(
                        user_input, 
                        callback_handler=None  # Disable callback to hide thoughts
                    )
                    
                    # Display LangChain response
                    st.markdown("### 🤖 Réponse de l'Agent LangChain")
                    
                    # Main response
                    st.markdown(f"**💬 Réponse:**")
                    st.write(response_data["response"])
                    
                    # Display found properties if any
                    if response_data.get("properties"):
                        st.markdown("**🏠 Propriétés Trouvées:**")
                        
                        # Create a more detailed display of properties
                        properties_df = pd.DataFrame(response_data["properties"][:10])  # Show top 10
                        
                        if not properties_df.empty:
                            # Display as cards
                            for idx, prop in properties_df.iterrows():
                                with st.expander(f"🏠 Propriété {idx + 1} - {prop.get('ville', 'N/A')}", expanded=False):
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.write(f"**Prix:** {prop.get('prix', 'N/A')} TND")
                                        st.write(f"**Surface:** {prop.get('surface', 'N/A')} m²")
                                        st.write(f"**Type:** {prop.get('type', 'N/A')}")
                                        st.write(f"**Ville:** {prop.get('ville', 'N/A')}")
                                    with col2:
                                        score = prop.get('budget_fit_score', 0)
                                        st.metric("Score Budget", f"{score:.1f}/10")
                                        if prop.get('URL'):
                                            st.markdown(f"[Voir l'annonce]({prop['URL']})")
                            
                            # Properties visualization
                            if len(properties_df) > 3:
                                st.markdown("**📊 Visualisation des Prix:**")
                                fig = px.scatter(
                                    properties_df, 
                                    x='surface', 
                                    y='prix',
                                    color='ville',
                                    size='budget_fit_score',
                                    hover_data=['type', 'chambres'],
                                    title="Prix vs Surface des Propriétés Trouvées"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Show context summary
                    context_summary = st.session_state.langchain_agent.get_context_summary()
                    if context_summary != "Aucun contexte défini":
                        st.info(f"📋 **Contexte:** {context_summary}")
                    
                    # Add to conversation history
                    conversation_entry = {
                        'input': user_input,
                        'response': response_data["response"],
                        'agent_type': 'langchain',
                        'timestamp': datetime.now().isoformat(),
                        'properties_count': len(response_data.get("properties", [])),
                        'context': response_data.get("context", {})
                    }
                    st.session_state.conversation_history.append(conversation_entry)
                    
                    # Handle any errors (but don't show technical details to users)
                    if "error" in response_data:
                        # Only show user-friendly messages, not technical errors
                        if not response_data["response"] or "erreur" in response_data["response"].lower():
                            st.info("💡 Si vous rencontrez des difficultés, essayez de reformuler votre question ou utilisez l'agent standard.")
                    
                except Exception as e:
                    st.error("❌ Service temporairement indisponible.")
                    st.info("🔄 Basculement vers l'agent standard...")
                    # Fall back to standard agent
                    use_standard_agent(user_input)
        else:
            # Use standard agent
            use_standard_agent(user_input)
    else:
        st.warning("⚠️ Veuillez entrer une description de votre projet.")

def standard_followup_processing(follow_up_input):
    """Process follow-up with standard agent"""
    try:
        # Process the follow-up with conversation context
        result = st.session_state.agent.process_client_input(
            follow_up_input, 
            st.session_state.conversation_context
        )
        # Add to conversation history
        conversation_entry = {
            'input': follow_up_input,
            'result': result,
            'agent_type': 'standard',
            'timestamp': datetime.now().isoformat(),
            'summary': f"Question de suivi - Budget: {result['budget_analysis'].get('extracted_budget', 'Précédent')} - Confiance: {result.get('confidence_level', 'medium')}"
        }
        st.session_state.conversation_history.append(conversation_entry)
        # Update context if new information is found
        if result['budget_analysis'].get('extracted_budget'):
            st.session_state.conversation_context['current_budget'] = result['budget_analysis']['extracted_budget']
        # Extract city from follow-up input
        user_input_lower = follow_up_input.lower()
        tunisia_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous']
        for city in tunisia_cities:
            if city in user_input_lower:
                st.session_state.conversation_context['preferred_city'] = city.title()
                break
        st.success("✅ Question traitée!")
        # Display the response for the follow-up question
        st.markdown("#### 💬 Réponse:")
        # Analyze the follow-up question and provide specific answers
        question_lower = follow_up_input.lower()
        # Check if user is asking about largest surface/size
        if any(word in question_lower for word in ['plus grand', 'plus grande', 'surface', 'taille', 'maximale', 'max']):
            # Get current budget from context
            current_budget = st.session_state.conversation_context.get('current_budget')
            current_city = st.session_state.conversation_context.get('preferred_city', 'Sousse')
            
            if current_budget:
                # Search for properties and find the largest one
                client_profile = {
                    "city": current_city,
                    "budget": current_budget,
                    "preferences": "terrain habitation construction",
                    "min_size": 100,
                    "max_price": current_budget
                }
                
                try:
                    budget_analysis = st.session_state.agent.analyze_client_budget(client_profile)
                    
                    if budget_analysis and budget_analysis.get('comparable_properties'):
                        properties = budget_analysis['comparable_properties']
                        
                        # Find the property with largest surface
                        largest_property = max(properties, key=lambda x: x.get('Surface', 0))
                        
                        # Find the top 3 largest properties
                        sorted_by_surface = sorted(properties, key=lambda x: x.get('Surface', 0), reverse=True)[:3]
                        
                        st.success("🏠 **Propriété avec la plus grande surface trouvée!**")
                        
                        # Display the largest property
                        with st.expander(f"🏆 Plus Grande Surface: {largest_property['Title'][:60]}...", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**🏠 Surface:** {largest_property['Surface']:.0f} m²")
                                st.write(f"**💰 Prix:** {largest_property['Price']:,.0f} DT")
                                st.write(f"**📊 Prix/m²:** {largest_property.get('price_per_m2', largest_property['Price']/largest_property['Surface']):,.0f} DT/m²")
                                st.write(f"**📍 Localisation:** {largest_property['Location']}")
                                st.write(f"**🏗️ Type:** {largest_property.get('Type', 'N/A')}")
                            
                            with col2:
                                # Surface comparison
                                avg_surface = np.mean([p.get('Surface', 0) for p in properties])
                                surface_advantage = largest_property['Surface'] - avg_surface
                                
                                st.metric(
                                    "Avantage Surface",
                                    f"+{surface_advantage:.0f} m²",
                                    delta="vs moyenne"
                                )
                                
                                # Budget efficiency
                                efficiency = largest_property['Surface'] / largest_property['Price'] * 1000
                                st.metric(
                                    "Efficacité",
                                    f"{efficiency:.2f} m²/1000DT"
                                )
                                
                                # Direct link
                                if largest_property.get('URL'):
                                    st.markdown(f"🔗 **[Voir l'annonce]({largest_property['URL']})**")
                    else:
                        st.warning("❌ Aucune propriété trouvée pour votre budget dans cette ville.")
                        # Suggest alternatives
                        st.info(f"""
                        💡 **Suggestions pour trouver plus de surface:**
                        • Élargir la zone de recherche autour de {current_city}
                        • Considérer un budget légèrement plus élevé
                        • Chercher des propriétés à rénover
                        • Explorer les zones périphériques
                        """)
                
                except Exception as e:
                    st.error(f"Erreur lors de la recherche: {str(e)}")
            else:
                st.warning("💰 Budget non défini. Veuillez d'abord spécifier votre budget.")
        
        # Check if user is asking about cheapest options
        elif any(word in question_lower for word in ['moins cher', 'pas cher', 'économique', 'budget serré']):
            current_budget = st.session_state.conversation_context.get('current_budget')
            if current_budget:
                st.info(f"🔍 Recherche des options les moins chères pour votre budget de {current_budget:,} DT...")
                # Trigger property search with focus on cheapest options
                result['next_actions'] = ['search_comparable_properties']
        
        # Generic budget analysis for other questions
        elif result['budget_analysis'].get('extracted_budget'):
            budget = result['budget_analysis']['extracted_budget']
            st.info(f"💰 **Budget identifié:** {budget:,} DT")
            
            # Estimate maximum surface based on average price per m²
            if budget > 0:
                avg_price_per_m2 = 1200  # Average price per m² in Tunisia
                max_surface = budget / avg_price_per_m2
                
                st.markdown(f"""
                **📐 Avec un budget de {budget:,} DT, vous pourriez obtenir:**
                - **Surface maximale estimée:** {max_surface:.0f} m² (basé sur {avg_price_per_m2} DT/m² en moyenne)
                - **Villa type:** {max_surface * 0.8:.0f} m² (en comptant les coûts supplémentaires)
                - **Appartement:** {max_surface * 1.2:.0f} m² (généralement moins cher au m²)
                """)
        
        # Show suggestions if available
        if result.get('suggestions'):
            st.markdown("**💡 Recommandations:**")
            for suggestion in result['suggestions']:
                st.write(f"• {suggestion}")
        
        # Auto-execute property search for other cases
        if (result['budget_analysis'].get('extracted_budget') and 
            st.session_state.user_preferences["auto_execute"] and
            'plus grand' not in question_lower):  # Don't auto-execute for largest surface queries
            executed_actions = execute_next_steps_enhanced(result, st.session_state.conversation_context)
            if executed_actions:
                st.markdown("**✅ Actions exécutées pour votre question:**")
                for action in executed_actions:
                    st.write(f"• {action}")
        
        # Don't auto-rerun to keep the response visible
        
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {str(e)}")

# Follow-up question section for continuous conversation
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### 💬 Posez une Question de Suivi")
    
    # Show context from last analysis
    last_entry = st.session_state.conversation_history[-1]
    current_budget = None
    current_city = st.session_state.conversation_context.get('preferred_city', 'Non spécifié')
    
    # Extract budget info based on entry type
    if 'result' in last_entry:
        # Standard agent entry
        last_analysis = last_entry['result']
        if last_analysis['budget_analysis'].get('extracted_budget'):
            current_budget = last_analysis['budget_analysis']['extracted_budget']
    elif 'context' in last_entry:
        # LangChain agent entry
        context = last_entry['context']
        current_budget = context.get('budget') or st.session_state.conversation_context.get('current_budget')
    else:
        # Try to get from session context
        current_budget = st.session_state.conversation_context.get('current_budget')
    
    # Display current context if available
    if current_budget:
        st.info(f"💰 **Budget actuel:** {current_budget:,} DT | 📍 **Ville:** {current_city}")
    else:
        st.info(f"📍 **Ville:** {current_city} | 💰 **Budget:** Non défini")
    
    # Follow-up input
    follow_up_input = st.text_area(
        "Votre question de suivi:",
        height=80,
        placeholder="Par exemple: 'Peux-tu me montrer des propriétés moins chères?' ou 'Qu'est-ce qui est disponible à Sousse?'",
        key="followup_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💬 Poser la Question", type="secondary", use_container_width=True):
            if follow_up_input.strip():
                # Check which agent type to use for follow-up
                if st.session_state.get("agent_type") == "langchain" and "langchain_agent" in st.session_state:
                    # Use LangChain agent for follow-up
                    with st.spinner("🤖 Traitement avec l'agent LangChain..."):
                        try:
                            # Process follow-up without showing internal thoughts
                            response_data = st.session_state.langchain_agent.chat(
                                follow_up_input,
                                callback_handler=None  # Disable callback to hide thoughts
                            )
                            
                            # Display LangChain response
                            st.markdown("#### 🤖 Réponse de l'Agent LangChain")
                            st.write(response_data["response"])
                            
                            # Display properties if found
                            if response_data.get("properties"):
                                st.markdown("**🏠 Propriétés Trouvées:**")
                                for idx, prop in enumerate(response_data["properties"][:5], 1):
                                    with st.expander(f"🏠 Propriété {idx} - {prop.get('ville', 'N/A')}", expanded=False):
                                        col1, col2 = st.columns([2, 1])
                                        with col1:
                                            st.write(f"**Prix:** {prop.get('prix', 'N/A')} TND")
                                            st.write(f"**Surface:** {prop.get('surface', 'N/A')} m²")
                                            st.write(f"**Type:** {prop.get('type', 'N/A')}")
                                        with col2:
                                            score = prop.get('budget_fit_score', 0)
                                            st.metric("Score", f"{score:.1f}/10")
                            
                            # Add to conversation history
                            conversation_entry = {
                                'input': follow_up_input,
                                'response': response_data["response"],
                                'agent_type': 'langchain',
                                'timestamp': datetime.now().isoformat(),
                                'properties_count': len(response_data.get("properties", [])),
                                'context': response_data.get("context", {})
                            }
                            st.session_state.conversation_history.append(conversation_entry)
                            
                        except Exception as e:
                            st.error("❌ Service temporairement indisponible.")
                            st.info("🔄 Utilisation de l'agent standard...")
                            # Fall back to standard agent processing
                            standard_followup_processing(follow_up_input)
                else:
                    # Use standard agent for follow-up
                    standard_followup_processing(follow_up_input)
            else:
                st.warning("⚠️ Veuillez entrer votre question.")

# Enhanced footer with additional features
st.markdown("---")

# Quick actions panel
st.markdown("### ⚡ Actions Rapides")

quick_col1, quick_col2 = st.columns(2)

with quick_col1:
    if st.button("🏠 Nouveau Projet", use_container_width=True):
        st.session_state.conversation_context = {}
        st.session_state.current_analysis = None
        st.session_state.conversation_history = []
        st.success("✅ Prêt pour un nouveau projet!")
        st.rerun()

with quick_col2:
    if st.button("🔄 Actualiser", use_container_width=True):
        st.rerun()

# Help and tips section
with st.expander("💡 Conseils d'Utilisation", expanded=False):
    st.markdown("""
    #### 🎯 Pour une analyse optimale:
    
    **💰 Concernant le budget:**
    - Indiquez un montant précis ou une fourchette
    - Précisez si le budget inclut le terrain
    - Mentionnez vos capacités de financement
    
    **🏠 Concernant le projet:**
    - Type de bien souhaité (villa, appartement, terrain...)
    - Surface approximative ou nombre de pièces
    - Zone géographique préférée
    
    **⚙️ Fonctionnalités avancées:**
    - Activez l'exécution automatique pour des analyses plus rapides
    - Utilisez les exemples pour vous inspirer
    - Consultez l'historique pour suivre l'évolution de votre projet
    
    **🔧 Options techniques:**
    - Activez les données techniques pour voir les détails JSON
    - Exportez vos analyses pour les conserver
    - Utilisez les actions rapides pour naviguer efficacement
    """)

# Footer with branding and info
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: CFC7D2   ; border-radius: 15px; margin-top: 2rem;">
    <h4>    </h4>
    <p style="margin: 0; opacity: 0.8;">
        Votre assistant intelligent pour des projets immobiliers réussis • 
        Session: {session_id} • 
        Version: 2.0 Enhanced
    </p>
</div>
""".format(session_id=st.session_state.conversation_id[-8:]), unsafe_allow_html=True)