import sys
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root_dir, '.env'))
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables should be set manually.")

# Debug environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

print(f"🔧 Environment Debug:")
print(f"   Root dir: {root_dir}")
print(f"   .env path: {os.path.join(root_dir, '.env')}")
print(f"   .env exists: {os.path.exists(os.path.join(root_dir, '.env'))}")
print(f"   GROQ_API_KEY loaded: {'✅' if groq_api_key else '❌'}")
print(f"   LANGSMITH_API_KEY loaded: {'✅' if langsmith_api_key else '❌'}")

import streamlit as st
import pandas as pd
import numpy as np

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Charts will be disabled.")

# Simple Budget Agent Implementation
class SimpleBudgetAgent:
    def __init__(self):
        self.tunisia_cities = [
            'tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 
            'monastir', 'nabeul', 'ariana', 'ben arous', 'gafsa', 'medenine',
            'jendouba', 'tataouine', 'tozeur', 'siliana', 'kasserine'
        ]
        
        # Mock property data for demonstration
        self.sample_properties = self._generate_sample_properties()
    
    def _generate_sample_properties(self) -> List[Dict]:
        """Generate sample property data for demonstration"""
        properties = []
        cities = ['Tunis', 'Sfax', 'Sousse', 'Monastir', 'Nabeul']
        
        for i in range(50):
            city = np.random.choice(cities)
            base_price = np.random.randint(80000, 500000)
            surface = np.random.randint(100, 1000)
            
            properties.append({
                'Title': f'Terrain {i+1} - {city}',
                'Price': base_price,
                'Surface': surface,
                'Location': f'{city}, Tunisie',
                'URL': f'https://example.com/property-{i+1}',
                'Type': 'Terrain',
                'City': city
            })
        
        return properties
    
    def extract_budget_info(self, text: str) -> Dict[str, Any]:
        """Extract budget information from user text"""
        result = {
            'extracted_budget': None,
            'budget_range': None,
            'city': None,
            'property_type': 'terrain',
            'confidence': 0.0
        }
        
        text_lower = text.lower()
        
        # Extract budget using regex
        budget_patterns = [
            r'(\d+(?:\s*\d+)*)\s*(?:mille|k)\s*dt?',  # 300 mille DT
            r'(\d+(?:\s*\d+)*)\s*dt?',                # 300000 DT
            r'budget\s*(?:de|:)?\s*(\d+(?:\s*\d+)*)',  # budget de 300000
            r'(\d+(?:\s*\d+)*)\s*dinars?',            # 300000 dinars
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                budget_str = matches[0].replace(' ', '')
                try:
                    budget = int(budget_str)
                    if 'mille' in text_lower or 'k' in text_lower:
                        budget *= 1000
                    
                    if 10000 <= budget <= 2000000:  # Reasonable range
                        result['extracted_budget'] = budget
                        result['confidence'] = 0.8
                        break
                except ValueError:
                    continue
        
        # Extract city
        for city in self.tunisia_cities:
            if city in text_lower:
                result['city'] = city.title()
                result['confidence'] += 0.1
                break
        
        # Extract budget range
        if 'entre' in text_lower or 'et' in text_lower:
            range_pattern = r'(\d+(?:\s*\d+)*)\s*(?:et|à)\s*(\d+(?:\s*\d+)*)'
            range_matches = re.findall(range_pattern, text_lower)
            if range_matches:
                try:
                    min_budget = int(range_matches[0][0].replace(' ', ''))
                    max_budget = int(range_matches[0][1].replace(' ', ''))
                    result['budget_range'] = (min_budget, max_budget)
                    result['extracted_budget'] = (min_budget + max_budget) // 2
                    result['confidence'] = 0.7
                except ValueError:
                    pass
        
        return result
    
    def search_properties(self, budget: int, city: str = None, max_results: int = 10) -> List[Dict]:
        """Search for properties based on budget and city"""
        filtered_properties = []
        
        for prop in self.sample_properties:
            # Filter by budget (±20% tolerance)
            if budget * 0.8 <= prop['Price'] <= budget * 1.2:
                # Filter by city if specified
                if city is None or prop['City'].lower() == city.lower():
                    filtered_properties.append(prop)
        
        # Sort by price (closest to budget first)
        filtered_properties.sort(key=lambda p: abs(p['Price'] - budget))
        
        return filtered_properties[:max_results]
    
    def analyze_client_budget(self, client_profile: Dict) -> Dict:
        """Analyze client budget and return market analysis"""
        budget = client_profile.get('budget', 0)
        city = client_profile.get('city', 'Sousse')
        
        # Find matching properties
        properties = self.search_properties(budget, city)
        
        # Calculate market statistics
        if properties:
            prices = [p['Price'] for p in properties]
            surfaces = [p['Surface'] for p in properties]
            
            market_stats = {
                'inventory_count': len(properties),
                'price_stats': {
                    'mean': np.mean(prices),
                    'median': np.median(prices),
                    'min': np.min(prices),
                    'max': np.max(prices)
                },
                'surface_stats': {
                    'mean': np.mean(surfaces),
                    'median': np.median(surfaces)
                },
                'budget_feasibility': {
                    'feasibility_ratio': len([p for p in properties if p['Price'] <= budget]) / len(properties)
                }
            }
        else:
            market_stats = {
                'inventory_count': 0,
                'price_stats': {'mean': 0, 'median': 0, 'min': 0, 'max': 0},
                'surface_stats': {'mean': 0, 'median': 0},
                'budget_feasibility': {'feasibility_ratio': 0}
            }
        
        return {
            'comparable_properties': properties,
            'market_statistics': market_stats,
            'analysis_date': datetime.now().isoformat()
        }
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and return comprehensive response"""
        # Extract budget information
        budget_info = self.extract_budget_info(message)
        
        # Generate response based on extracted information
        response_parts = []
        
        if budget_info['extracted_budget']:
            budget = budget_info['extracted_budget']
            response_parts.append(f"💰 **Budget détecté:** {budget:,} DT")
            
            if budget_info['city']:
                response_parts.append(f"📍 **Ville:** {budget_info['city']}")
            
            # Provide budget analysis
            if budget < 100000:
                response_parts.append("⚠️ **Analyse:** Budget limité. Je recommande de chercher dans les zones périphériques ou d'augmenter le budget.")
            elif budget < 200000:
                response_parts.append("✅ **Analyse:** Budget correct pour des terrains de taille moyenne.")
            else:
                response_parts.append("🎯 **Analyse:** Excellent budget! Vous avez plusieurs options intéressantes.")
        
        else:
            response_parts.append("🤔 Je n'ai pas pu identifier votre budget précis. Pouvez-vous me dire combien vous souhaitez investir ?")
        
        # Generate suggestions
        suggestions = []
        if budget_info['extracted_budget']:
            suggestions.append("Rechercher des propriétés dans votre budget")
            suggestions.append("Analyser le marché local")
            suggestions.append("Comparer avec d'autres régions")
        else:
            suggestions.append("Préciser votre budget")
            suggestions.append("Indiquer la ville d'intérêt")
            suggestions.append("Spécifier la surface souhaitée")
        
        return {
            'agent_response': '\n\n'.join(response_parts),
            'budget_analysis': budget_info,
            'suggestions': suggestions,
            'reliability_score': budget_info['confidence']
        }

# Try to import LangChain agent
try:
    from langchain_budget_agent import create_langchain_budget_agent
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain agent not available: {e}")
    LANGCHAIN_AVAILABLE = False

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
    
    .property-card {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 Agent Budget Immobilier</h1>
    <p>Votre assistant intelligent pour l'analyse budgétaire et la recherche de propriétés</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "agent": None,
        "messages": [],
        "conversation_context": [],
        "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "agent_type": "simple"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Initialize agent
@st.cache_resource
def initialize_agent():
    """Initialize the budget agent"""
    try:
        # Try LangChain agent first if available
        if groq_api_key and LANGCHAIN_AVAILABLE:
            try:
                print("🔄 Attempting to initialize LangChain Budget Agent...")
                agent = create_langchain_budget_agent()
                st.success("✅ LangChain Budget Agent initialized!")
                return agent, "langchain"
            except Exception as e:
                st.warning(f"⚠️ LangChain agent failed: {str(e)}")
                print(f"LangChain initialization error: {e}")
        
        # Fallback to simple agent
        print("🔄 Initializing Simple Budget Agent...")
        agent = SimpleBudgetAgent()
        st.info("✅ Simple Budget Agent initialized!")
        return agent, "simple"
        
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None, "none"

# Initialize agent if not already done
if st.session_state.agent is None:
    with st.spinner("🔄 Initialisation de l'agent budget..."):
        agent, agent_type = initialize_agent()
        st.session_state.agent = agent
        st.session_state.agent_type = agent_type

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Tableau de Bord")
    
    # Agent status
    if st.session_state.agent_type == "langchain":
        st.success("🤖 **Agent LangChain** - IA Avancée")
    elif st.session_state.agent_type == "simple":
        st.info("🤖 **Agent Simple** - Analyse rapide")
    else:
        st.error("❌ **Agent non disponible**")
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Statistiques")
    if st.session_state.messages:
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Messages utilisateur", user_messages)
        st.metric("Réponses agent", len(st.session_state.messages) - user_messages)
    
    st.divider()
    
    # Clear conversation
    if st.button("🗑️ Nouvelle Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

# Property card renderer
def render_property_card(prop: Dict, index: int):
    """Render a property card"""
    price_per_m2 = prop['Price'] / prop['Surface'] if prop['Surface'] > 0 else 0
    
    st.markdown(f"""
    <div class="property-card">
        <h4>🏠 {prop['Title']}</h4>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <span><strong>💰 Prix:</strong> {prop['Price']:,} DT</span>
            <span><strong>📐 Surface:</strong> {prop['Surface']} m²</span>
            <span><strong>📊 Prix/m²:</strong> {price_per_m2:,.0f} DT/m²</span>
        </div>
        <p><strong>📍 Localisation:</strong> {prop['Location']}</p>
        {f'<a href="{prop["URL"]}" target="_blank" style="color: #007bff;">🔗 Voir l\'annonce</a>' if prop.get('URL') and prop['URL'] != 'No URL available' else ''}
    </div>
    """, unsafe_allow_html=True)

# Initialize conversation
if not st.session_state.messages:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis votre assistant immobilier. Je peux vous aider à analyser votre budget et rechercher des propriétés. Comment puis-je vous aider aujourd'hui ?",
        "timestamp": datetime.now().isoformat()
    }]

# Display conversation
st.markdown("### 💬 Conversation")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 10px 15px; border-radius: 18px 18px 5px 18px; 
                        max-width: 70%; margin-left: 30%;">
                <strong>Vous:</strong><br>{message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background: #f0f2f6; color: #262730; padding: 10px 15px; 
                        border-radius: 18px 18px 18px 5px; max-width: 70%; margin-right: 30%;">
                <strong>🤖 Assistant:</strong><br>{message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display properties if available
        if message.get('properties'):
            st.markdown("### 🏠 Propriétés Trouvées")
            for i, prop in enumerate(message['properties'], 1):
                render_property_card(prop, i)

# Chat input
def process_user_message(user_input: str):
    """Process user message and generate response"""
    try:
        if st.session_state.agent_type == "langchain":
            # Use LangChain agent
            try:
                if hasattr(st.session_state.agent, 'invoke'):
                    result = st.session_state.agent.invoke({"input": user_input})
                    response = result.get("output", "Désolé, je n'ai pas pu traiter votre demande.")
                else:
                    response = st.session_state.agent.run(user_input)
                
                return {
                    'response': response,
                    'properties': [],
                    'should_search': 'cherche' in user_input.lower() or 'trouve' in user_input.lower()
                }
            except Exception as e:
                return {
                    'response': f"❌ Erreur avec l'agent LangChain: {str(e)}",
                    'properties': [],
                    'should_search': False
                }
        
        else:
            # Use simple agent
            result = st.session_state.agent.process_message(user_input)
            
            response = result['agent_response']
            properties = []
            
            # Check if we should search for properties
            budget_info = result.get('budget_analysis', {})
            if (budget_info.get('extracted_budget') and 
                ('cherche' in user_input.lower() or 'trouve' in user_input.lower() or 
                 'propriété' in user_input.lower() or 'terrain' in user_input.lower())):
                
                # Search for properties
                client_profile = {
                    'budget': budget_info['extracted_budget'],
                    'city': budget_info.get('city', 'Sousse')
                }
                
                analysis = st.session_state.agent.analyze_client_budget(client_profile)
                properties = analysis['comparable_properties'][:5]
                
                if properties:
                    response += f"\n\n🏠 **J'ai trouvé {len(properties)} propriétés correspondant à votre budget!**"
                else:
                    response += "\n\n😔 **Aucune propriété trouvée dans votre budget. Voulez-vous élargir les critères?**"
            
            return {
                'response': response,
                'properties': properties,
                'should_search': False
            }
    
    except Exception as e:
        return {
            'response': f"❌ Erreur lors du traitement: {str(e)}",
            'properties': [],
            'should_search': False
        }

# Input area
st.markdown("---")
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Tapez votre message...",
        placeholder="Ex: J'ai un budget de 300000 DT pour un terrain à Sousse",
        key="chat_input"
    )

with col2:
    send_button = st.button("📤", help="Envoyer", use_container_width=True)

# Handle message sending
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process with agent
    with st.spinner("🤖 L'assistant réfléchit..."):
        result = process_user_message(user_input)
        
        # Add assistant response
        assistant_message = {
            "role": "assistant",
            "content": result['response'],
            "timestamp": datetime.now().isoformat()
        }
        
        if result['properties']:
            assistant_message['properties'] = result['properties']
        
        st.session_state.messages.append(assistant_message)
    
    # Clear input and refresh
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <p>🏗️ <strong>Agent Budget Immobilier</strong> - Session: {st.session_state.conversation_id[-8:]}</p>
    <p style="font-size: 0.9em; color: #666;">Votre assistant intelligent pour des projets immobiliers réussis</p>
</div>
""", unsafe_allow_html=True)

# Help section
with st.expander("💡 Guide d'Utilisation"):
    st.markdown("""
    #### Comment utiliser l'assistant:
    
    **💰 Pour l'analyse budgétaire:**
    - Mentionnez votre budget: "J'ai 250000 DT"
    - Précisez la ville: "pour un terrain à Sousse"
    - L'assistant analysera automatiquement vos capacités
    
    **🔍 Pour la recherche:**
    - Utilisez des mots comme "cherche", "trouve", "propriété"
    - Soyez précis sur vos critères
    - L'assistant trouvera les meilleures options
    
    **📊 Fonctionnalités:**
    - Analyse de budget automatique
    - Recherche de propriétés
    - Statistiques du marché
    - Conseils personnalisés
    """)