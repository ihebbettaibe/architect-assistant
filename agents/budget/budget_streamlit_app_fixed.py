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

# Enhanced Budget Agent Implementation with Memory
class SimpleBudgetAgent:
    def __init__(self):
        self.tunisia_cities = [
            'tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 'mahdia', 
            'monastir', 'nabeul', 'ariana', 'ben arous', 'gafsa', 'medenine',
            'jendouba', 'tataouine', 'tozeur', 'siliana', 'kasserine'
        ]
        
        # Conversation memory and context
        self.conversation_memory = {
            'user_profile': {},
            'search_history': [],
            'preferences': {},
            'last_budget': None,
            'last_city': None,
            'interaction_count': 0
        }
        
        # Enhanced property data with real-looking URLs
        self.sample_properties = self._generate_enhanced_properties()
    
    def _generate_enhanced_properties(self) -> List[Dict]:
        """Generate enhanced property data with realistic details and URLs"""
        properties = []
        cities = ['Tunis', 'Sfax', 'Sousse', 'Monastir', 'Nabeul', 'Bizerte', 'Mahdia']
        property_types = ['Terrain', 'Villa', 'Appartement', 'Maison']
        
        # Real estate websites for Tunisia
        websites = [
            'tayara.tn', 'jumia.com.tn', 'afariat.com', 'immobilier.tn',
            'mubawab.tn', 'tecnocasa.tn', 'sarouty.tn'
        ]
        
        neighborhoods = {
            'Tunis': ['La Marsa', 'Carthage', 'Sidi Bou Said', 'Ariana', 'Manouba'],
            'Sfax': ['Centre Ville', 'Sakiet Ezzit', 'Route Tunis', 'Sfax Sud'],
            'Sousse': ['Centre Ville', 'Port El Kantaoui', 'Hammam Sousse', 'Sahloul'],
            'Monastir': ['Centre Ville', 'Skanes', 'Ksar Hellal', 'Jemmal'],
            'Nabeul': ['Centre Ville', 'Hammamet', 'Kelibia', 'Korba'],
            'Bizerte': ['Centre Ville', 'Zarzouna', 'Ras Jebel', 'Mateur'],
            'Mahdia': ['Centre Ville', 'Hiboun', 'Ksour Essef', 'Chorbane']
        }
        
        for i in range(100):
            city = np.random.choice(cities)
            neighborhood = np.random.choice(neighborhoods.get(city, ['Centre Ville']))
            prop_type = np.random.choice(property_types)
            website = np.random.choice(websites)
            
            # More realistic price ranges based on city and type
            base_prices = {
                'Tunis': {'Terrain': (120000, 800000), 'Villa': (300000, 1200000), 
                         'Appartement': (80000, 500000), 'Maison': (150000, 700000)},
                'Sfax': {'Terrain': (80000, 400000), 'Villa': (200000, 800000), 
                        'Appartement': (60000, 350000), 'Maison': (100000, 450000)},
                'Sousse': {'Terrain': (100000, 600000), 'Villa': (250000, 900000), 
                          'Appartement': (70000, 400000), 'Maison': (120000, 550000)},
                'Monastir': {'Terrain': (90000, 500000), 'Villa': (220000, 750000), 
                            'Appartement': (65000, 380000), 'Maison': (110000, 500000)},
                'Nabeul': {'Terrain': (70000, 350000), 'Villa': (180000, 600000), 
                          'Appartement': (50000, 280000), 'Maison': (90000, 400000)},
                'Bizerte': {'Terrain': (60000, 300000), 'Villa': (150000, 500000), 
                           'Appartement': (40000, 250000), 'Maison': (80000, 350000)},
                'Mahdia': {'Terrain': (65000, 320000), 'Villa': (160000, 520000), 
                          'Appartement': (45000, 260000), 'Maison': (85000, 370000)}
            }
            
            price_range = base_prices.get(city, base_prices['Sousse'])[prop_type]
            price = np.random.randint(price_range[0], price_range[1])
            
            # Surface based on property type
            surface_ranges = {
                'Terrain': (200, 2000),
                'Villa': (150, 800),
                'Appartement': (60, 250),
                'Maison': (100, 500)
            }
            
            surface = np.random.randint(*surface_ranges[prop_type])
            
            # Generate realistic property ID
            prop_id = f"{city[:3].upper()}{prop_type[:2].upper()}{i+1:03d}"
            
            properties.append({
                'ID': prop_id,
                'Title': f'{prop_type} {surface}m² à {neighborhood}, {city}',
                'Price': price,
                'Surface': surface,
                'Location': f'{neighborhood}, {city}, Tunisie',
                'URL': f'https://www.{website}/annonces/{prop_id.lower()}-{prop_type.lower()}-{city.lower()}',
                'Type': prop_type,
                'City': city,
                'Neighborhood': neighborhood,
                'Description': self._generate_property_description(prop_type, surface, neighborhood, city),
                'Features': self._generate_property_features(prop_type),
                'Contact': f'+216 {np.random.randint(20000000, 99999999)}',
                'Posted_Date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d')
            })
        
        return properties
    
    def _generate_property_description(self, prop_type: str, surface: int, neighborhood: str, city: str) -> str:
        """Generate realistic property descriptions"""
        descriptions = {
            'Terrain': [
                f"Excellent terrain de {surface}m² situé dans le quartier prisé de {neighborhood}.",
                f"Terrain constructible de {surface}m² avec toutes les commodités à proximité.",
                f"Belle opportunité d'investissement : terrain de {surface}m² bien exposé."
            ],
            'Villa': [
                f"Magnifique villa de {surface}m² avec jardin et piscine à {neighborhood}.",
                f"Villa moderne de {surface}m² entièrement rénovée, quartier calme.",
                f"Splendide villa de {surface}m² avec vue panoramique sur {city}."
            ],
            'Appartement': [
                f"Bel appartement de {surface}m² au cœur de {neighborhood}.",
                f"Appartement moderne de {surface}m² avec balcon et parking.",
                f"Charmant appartement de {surface}m² proche de toutes commodités."
            ],
            'Maison': [
                f"Maison familiale de {surface}m² avec cour et garage à {neighborhood}.",
                f"Belle maison de {surface}m² entièrement rénovée.",
                f"Maison traditionnelle de {surface}m² dans un quartier résidentiel."
            ]
        }
        return np.random.choice(descriptions.get(prop_type, ["Propriété à vendre"]))
    
    def _generate_property_features(self, prop_type: str) -> List[str]:
        """Generate property features based on type"""
        common_features = ["Proche écoles", "Transport public", "Commerces à proximité"]
        
        type_features = {
            'Terrain': ["Constructible", "Raccordé aux réseaux", "Bien exposé"],
            'Villa': ["Piscine", "Jardin", "Garage", "Terrasse"],
            'Appartement': ["Ascenseur", "Balcon", "Parking", "Climatisation"],
            'Maison': ["Cour", "Garage", "Terrasse", "Cave"]
        }
        
        features = common_features + type_features.get(prop_type, [])
        return np.random.choice(features, size=min(4, len(features)), replace=False).tolist()
    
    def update_conversation_memory(self, user_message: str, extracted_info: Dict):
        """Update conversation memory with new information"""
        self.conversation_memory['interaction_count'] += 1
        
        # Update user profile
        if extracted_info.get('extracted_budget'):
            self.conversation_memory['user_profile']['budget'] = extracted_info['extracted_budget']
            self.conversation_memory['last_budget'] = extracted_info['extracted_budget']
        
        if extracted_info.get('city'):
            self.conversation_memory['user_profile']['preferred_city'] = extracted_info['city']
            self.conversation_memory['last_city'] = extracted_info['city']
        
        # Add to search history
        search_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': user_message,
            'extracted_info': extracted_info
        }
        self.conversation_memory['search_history'].append(search_entry)
        
        # Keep only last 10 searches
        if len(self.conversation_memory['search_history']) > 10:
            self.conversation_memory['search_history'] = self.conversation_memory['search_history'][-10:]
    
    def get_contextual_response(self, message: str, extracted_info: Dict) -> str:
        """Generate contextual response based on conversation history"""
        is_first_interaction = self.conversation_memory['interaction_count'] == 0
        has_previous_budget = self.conversation_memory['last_budget'] is not None
        has_previous_city = self.conversation_memory['last_city'] is not None
        
        response_parts = []
        
        # Greeting for first interaction
        if is_first_interaction:
            response_parts.append("👋 **Bienvenue ! Je suis ravi de vous aider dans votre recherche immobilière.**")
        
        # Budget analysis
        if extracted_info.get('extracted_budget'):
            budget = extracted_info['extracted_budget']
            response_parts.append(f"💰 **Budget détecté:** {budget:,} DT")
            
            # Compare with previous budget if any
            if has_previous_budget and self.conversation_memory['last_budget'] != budget:
                prev_budget = self.conversation_memory['last_budget']
                if budget > prev_budget:
                    response_parts.append(f"📈 **Excellent !** Vous avez augmenté votre budget de {budget - prev_budget:,} DT")
                else:
                    response_parts.append(f"📉 **Nouveau budget** ajusté à la baisse de {prev_budget - budget:,} DT")
        elif has_previous_budget:
            # Use previous budget if none detected
            budget = self.conversation_memory['last_budget']
            response_parts.append(f"💰 **Budget précédent:** {budget:,} DT (je garde en mémoire)")
        
        # City analysis
        if extracted_info.get('city'):
            city = extracted_info['city']
            response_parts.append(f"📍 **Ville:** {city}")
            
            if has_previous_city and self.conversation_memory['last_city'] != city:
                response_parts.append(f"🗺️ **Changement de zone** de {self.conversation_memory['last_city']} vers {city}")
        elif has_previous_city:
            city = self.conversation_memory['last_city']
            response_parts.append(f"📍 **Ville précédente:** {city} (je continue sur cette zone)")
        
        # Budget assessment
        budget = extracted_info.get('extracted_budget') or self.conversation_memory['last_budget']
        if budget:
            if budget < 100000:
                response_parts.append("⚠️ **Analyse:** Budget serré. Je recommande les zones périphériques ou terrains plus petits.")
            elif budget < 200000:
                response_parts.append("✅ **Analyse:** Budget équilibré pour de belles opportunités.")
            elif budget < 500000:
                response_parts.append("🎯 **Analyse:** Excellent budget! Vous avez accès à des propriétés de qualité.")
            else:
                response_parts.append("💎 **Analyse:** Budget premium! Les meilleures propriétés s'offrent à vous.")
        
        # Conversation continuity
        if self.conversation_memory['interaction_count'] > 1:
            response_parts.append(f"🔄 **Suivi:** C'est notre {self.conversation_memory['interaction_count']}ème échange. Je garde tout en mémoire.")
        
        return '\n\n'.join(response_parts)
    
    def extract_budget_info(self, text: str, use_memory: bool = True) -> Dict[str, Any]:
        """Extract budget information from user text with memory context"""
        result = {
            'extracted_budget': None,
            'budget_range': None,
            'city': None,
            'property_type': 'terrain',
            'confidence': 0.0,
            'search_intent': False
        }
        
        text_lower = text.lower()
        
        # Check for search intent
        search_keywords = ['cherche', 'trouve', 'recherche', 'propriété', 'terrain', 'maison', 'villa', 'appartement', 'recommande', 'propose', 'montre']
        result['search_intent'] = any(keyword in text_lower for keyword in search_keywords)
        
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
        
        # Use memory if no budget found and memory is available
        if not result['extracted_budget'] and use_memory and self.conversation_memory['last_budget']:
            result['extracted_budget'] = self.conversation_memory['last_budget']
            result['confidence'] = 0.6  # Lower confidence for memory-based budget
        
        # Extract city
        for city in self.tunisia_cities:
            if city in text_lower:
                result['city'] = city.title()
                result['confidence'] += 0.1
                break
        
        # Use memory city if none found
        if not result['city'] and use_memory and self.conversation_memory['last_city']:
            result['city'] = self.conversation_memory['last_city']
            result['confidence'] += 0.05
        
        # Extract property type
        property_types = {
            'terrain': ['terrain', 'lot', 'parcelle'],
            'villa': ['villa', 'maison individuelle'],
            'appartement': ['appartement', 'appart', 'flat'],
            'maison': ['maison', 'house']
        }
        
        for prop_type, keywords in property_types.items():
            if any(keyword in text_lower for keyword in keywords):
                result['property_type'] = prop_type
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
    
    def search_properties(self, budget: int, city: str = None, property_type: str = None, max_results: int = 8) -> List[Dict]:
        """Enhanced property search with filtering and ranking"""
        filtered_properties = []
        
        for prop in self.sample_properties:
            # Filter by budget (±25% tolerance)
            if budget * 0.75 <= prop['Price'] <= budget * 1.25:
                # Filter by city if specified
                if city is None or prop['City'].lower() == city.lower():
                    # Filter by property type if specified
                    if property_type is None or prop['Type'].lower() == property_type.lower():
                        # Calculate match score
                        price_diff = abs(prop['Price'] - budget) / budget
                        match_score = 1 - price_diff
                        prop['match_score'] = match_score
                        prop['recommendation_reason'] = self._get_recommendation_reason(prop, budget, city)
                        filtered_properties.append(prop)
        
        # Sort by match score (best matches first)
        filtered_properties.sort(key=lambda p: p['match_score'], reverse=True)
        
        return filtered_properties[:max_results]
    
    def _get_recommendation_reason(self, prop: Dict, budget: int, city: str = None) -> str:
        """Generate recommendation reason for each property"""
        reasons = []
        
        price_diff_percent = abs(prop['Price'] - budget) / budget * 100
        
        if price_diff_percent < 5:
            reasons.append("💯 Prix parfaitement dans votre budget")
        elif price_diff_percent < 15:
            reasons.append("✅ Excellent rapport qualité-prix")
        elif prop['Price'] < budget:
            reasons.append(f"💰 Économie de {budget - prop['Price']:,} DT")
        
        # Surface-based recommendations
        surface_per_dt = prop['Surface'] / prop['Price']
        if surface_per_dt > 0.002:  # Good surface per DT ratio
            reasons.append("📐 Excellente surface pour le prix")
        
        # Location-based recommendations
        if city and prop['City'].lower() == city.lower():
            reasons.append("📍 Dans votre ville préférée")
        
        # Type-specific reasons
        if prop['Type'] == 'Villa':
            reasons.append("🏡 Idéal pour famille")
        elif prop['Type'] == 'Terrain':
            reasons.append("🏗️ Parfait pour construction")
        elif prop['Type'] == 'Appartement':
            reasons.append("🏢 Prêt à habiter")
        
        return ' • '.join(reasons[:3])  # Max 3 reasons
    
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
        """Enhanced message processing with conversation memory"""
        # Extract budget information with memory context
        budget_info = self.extract_budget_info(message, use_memory=True)
        
        # Update conversation memory
        self.update_conversation_memory(message, budget_info)
        
        # Generate contextual response
        response = self.get_contextual_response(message, budget_info)
        
        # Always search for properties when budget is available (default behavior)
        should_search = (
            budget_info.get('extracted_budget') is not None or
            budget_info.get('search_intent', False) or
            'cherche' in message.lower() or 
            'trouve' in message.lower() or 
            'recommande' in message.lower() or
            'propriété' in message.lower() or 
            'terrain' in message.lower() or
            'montre' in message.lower() or
            'propose' in message.lower()
        )
        
        # Enhanced suggestions based on context
        suggestions = self._generate_contextual_suggestions(budget_info, should_search)
        
        return {
            'agent_response': response,
            'budget_analysis': budget_info,
            'suggestions': suggestions,
            'should_search': should_search,
            'reliability_score': budget_info['confidence'],
            'conversation_context': self.conversation_memory
        }
    
    def _generate_contextual_suggestions(self, budget_info: Dict, should_search: bool) -> List[str]:
        """Generate contextual suggestions based on conversation state"""
        suggestions = []
        
        if budget_info.get('extracted_budget'):
            if should_search:
                suggestions.extend([
                    "Voir plus de propriétés similaires",
                    "Comparer avec d'autres villes",
                    "Analyser les tendances du marché"
                ])
            else:
                suggestions.extend([
                    "Rechercher des propriétés dans ce budget",
                    "Voir les options dans d'autres villes",
                    "Analyser le marché local"
                ])
        else:
            suggestions.extend([
                "Préciser votre budget disponible",
                "Indiquer votre ville d'intérêt",
                "Spécifier le type de propriété souhaité"
            ])
        
        # Add memory-based suggestions
        if self.conversation_memory['interaction_count'] > 1:
            suggestions.append("Revoir nos échanges précédents")
        
        if self.conversation_memory['search_history']:
            suggestions.append("Comparer avec vos recherches précédentes")
        
        return suggestions[:4]  # Limit to 4 suggestions

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
    
    # Enhanced conversation memory display
    st.subheader("🧠 Mémoire de Conversation")
    if hasattr(st.session_state.agent, 'conversation_memory') and st.session_state.agent:
        memory = st.session_state.agent.conversation_memory
        if memory['last_budget']:
            st.metric("Budget mémorisé", f"{memory['last_budget']:,} DT")
        if memory['last_city']:
            st.metric("Ville mémorisée", memory['last_city'])
        st.metric("Interactions", memory['interaction_count'])
    
    st.divider()
    
    # Clear conversation
    if st.button("🗑️ Nouvelle Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        # Reset agent memory
        if hasattr(st.session_state.agent, 'conversation_memory'):
            st.session_state.agent.conversation_memory = {
                'user_profile': {},
                'search_history': [],
                'preferences': {},
                'last_budget': None,
                'last_city': None,
                'interaction_count': 0
            }
        st.rerun()

# Enhanced Property card renderer
def render_property_card(prop: Dict, index: int):
    """Render an enhanced property card with full details"""
    price_per_m2 = prop['Price'] / prop['Surface'] if prop['Surface'] > 0 else 0
    
    # Property type icon
    type_icons = {
        'Terrain': '🏗️',
        'Villa': '🏡',
        'Appartement': '🏢',
        'Maison': '🏠'
    }
    
    icon = type_icons.get(prop['Type'], '🏘️')
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                border-radius: 15px; padding: 20px; margin: 15px 0; 
                border-left: 5px solid #007bff; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #2c3e50;">{icon} {prop['Title']}</h4>
            <span style="background: #007bff; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.9em;">
                #{index}
            </span>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 10px;">
                <div style="text-align: center;">
                    <strong style="color: #28a745;">💰 {prop['Price']:,} DT</strong>
                </div>
                <div style="text-align: center;">
                    <strong style="color: #17a2b8;">📐 {prop['Surface']} m²</strong>
                </div>
                <div style="text-align: center;">
                    <strong style="color: #fd7e14;">📊 {price_per_m2:,.0f} DT/m²</strong>
                </div>
            </div>
        </div>
        
        <div style="margin: 10px 0;">
            <p style="margin: 5px 0;"><strong>📍 Localisation:</strong> {prop['Location']}</p>
            <p style="margin: 5px 0; font-style: italic; color: #666;">{prop.get('Description', 'Belle propriété à découvrir')}</p>
        </div>
        
        {f'''<div style="margin: 10px 0;">
            <p style="margin: 5px 0;"><strong>✨ Caractéristiques:</strong></p>
            <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                {' '.join([f'<span style="background: #e7f3ff; color: #0066cc; padding: 3px 8px; border-radius: 12px; font-size: 0.85em;">{feature}</span>' for feature in prop.get('Features', [])])}
            </div>
        </div>''' if prop.get('Features') else ''}
        
        {f'''<div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 10px; margin: 10px 0;">
            <strong>🎯 Pourquoi cette propriété ?</strong><br>
            <span style="color: #856404;">{prop.get('recommendation_reason', 'Excellente opportunité')}</span>
        </div>''' if prop.get('recommendation_reason') else ''}
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px; padding-top: 10px; border-top: 1px solid #dee2e6;">
            <div style="font-size: 0.9em; color: #666;">
                � Publié le {prop.get('Posted_Date', 'récemment')}
            </div>
            <div style="display: flex; gap: 10px;">
                {f'<span style="background: #28a745; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">📞 {prop.get("Contact", "Contact disponible")}</span>' if prop.get('Contact') else ''}
                <a href="{prop.get('URL', '#')}" target="_blank" 
                   style="background: #007bff; color: white; padding: 8px 15px; border-radius: 20px; 
                          text-decoration: none; font-weight: bold; font-size: 0.9em;">
                    🔗 Voir l'annonce
                </a>
            </div>
        </div>
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
        
        # Display properties if available with enhanced info
        if message.get('properties'):
            st.markdown("### 🏠 Propriétés Recommandées")
            
            # Add summary statistics
            if len(message['properties']) > 0:
                avg_price = sum(p['Price'] for p in message['properties']) / len(message['properties'])
                min_price = min(p['Price'] for p in message['properties'])
                max_price = max(p['Price'] for p in message['properties'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔍 Trouvées", len(message['properties']))
                with col2:
                    st.metric("💰 Prix moyen", f"{avg_price:,.0f} DT")
                with col3:
                    st.metric("📉 Min", f"{min_price:,.0f} DT")
                with col4:
                    st.metric("📈 Max", f"{max_price:,.0f} DT")
                
                st.markdown("---")
            
            for i, prop in enumerate(message['properties'], 1):
                render_property_card(prop, i)
                
            if len(message['properties']) >= 8:
                st.info("💡 Affichage des 8 meilleures correspondances. Affinez vos critères pour plus de précision.")
        
        # Display suggestions if available
        if message.get('suggestions'):
            st.markdown("### 💡 Suggestions")
            cols = st.columns(min(len(message['suggestions']), 4))
            for i, suggestion in enumerate(message['suggestions'][:4]):
                with cols[i]:
                    if st.button(f"💭 {suggestion}", key=f"suggestion_{message['timestamp']}_{i}", use_container_width=True):
                        # Add suggestion as user message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": suggestion,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.rerun()

# Enhanced Chat input processing
def process_user_message(user_input: str):
    """Enhanced message processing with memory and property search"""
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
                    'suggestions': ["Préciser votre budget", "Indiquer la ville", "Type de propriété"],
                    'context_info': {}
                }
            except Exception as e:
                return {
                    'response': f"❌ Erreur avec l'agent LangChain: {str(e)}",
                    'properties': [],
                    'suggestions': [],
                    'context_info': {}
                }
        
        else:
            # Enhanced simple agent processing
            result = st.session_state.agent.process_message(user_input)
            
            response = result['agent_response']
            properties = []
            
            # Get budget and search info
            budget_info = result.get('budget_analysis', {})
            should_search = result.get('should_search', False)
            
            # Always search for properties when budget is detected (default behavior)
            if budget_info.get('extracted_budget'):
                client_profile = {
                    'budget': budget_info['extracted_budget'],
                    'city': budget_info.get('city'),
                    'property_type': budget_info.get('property_type', 'terrain')
                }
                
                # Enhanced property search
                properties = st.session_state.agent.search_properties(
                    budget=client_profile['budget'],
                    city=client_profile['city'],
                    property_type=client_profile['property_type'],
                    max_results=8
                )
                
                if properties:
                    # Always format property details first (default behavior)
                    property_summary = "\n\n� **PROPRIÉTÉS RECOMMANDÉES:**\n"
                    for i, prop in enumerate(properties[:3], 1):  # Show top 3 in summary
                        property_summary += f"\n**#{i} - {prop['Title']}**\n"
                        property_summary += f"📍 {prop['Location']}\n"
                        property_summary += f"💰 {prop['Price']:,} DT\n"
                        property_summary += f"🔗 {prop['URL']}\n"
                    
                    if len(properties) > 3:
                        property_summary += f"\n... et {len(properties) - 3} autres propriétés disponibles\n"
                    
                    response = property_summary + "\n" + response
                    response += f"\n\n🎯 **Total trouvé:** {len(properties)} propriétés adaptées à vos critères"
                else:
                    response += "\n\n😔 **Aucune propriété trouvée avec ces critères exacts.**"
                    response += "\n💭 **Suggestions:** Élargir le budget de ±20% ou changer de ville ?"
            
            return {
                'response': response,
                'properties': properties,
                'suggestions': result.get('suggestions', []),
                'context_info': {
                    'budget': budget_info.get('extracted_budget'),
                    'city': budget_info.get('city'),
                    'confidence': budget_info.get('confidence', 0),
                    'search_performed': bool(properties)
                }
            }
    
    except Exception as e:
        return {
            'response': f"❌ Erreur lors du traitement: {str(e)}",
            'properties': [],
            'suggestions': [],
            'context_info': {}
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

# Enhanced Handle message sending with context and suggestions
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process with enhanced agent
    with st.spinner("🤖 L'assistant analyse votre demande..."):
        result = process_user_message(user_input)
        
        # Add assistant response with enhanced data
        assistant_message = {
            "role": "assistant",
            "content": result['response'],
            "timestamp": datetime.now().isoformat(),
            "context_info": result.get('context_info', {})
        }
        
        if result['properties']:
            assistant_message['properties'] = result['properties']
        
        if result.get('suggestions'):
            assistant_message['suggestions'] = result['suggestions']
        
        st.session_state.messages.append(assistant_message)
    
    # Clear input and refresh
    st.rerun()

# Enhanced footer with conversation context
conversation_context = ""
if hasattr(st.session_state.agent, 'conversation_memory') and st.session_state.agent:
    memory = st.session_state.agent.conversation_memory
    if memory['last_budget'] or memory['last_city']:
        context_parts = []
        if memory['last_budget']:
            context_parts.append(f"Budget: {memory['last_budget']:,} DT")
        if memory['last_city']:
            context_parts.append(f"Ville: {memory['last_city']}")
        conversation_context = f" | Contexte: {' • '.join(context_parts)}"

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <p>🏗️ <strong>Agent Budget Immobilier</strong> - Session: {st.session_state.conversation_id[-8:]}{conversation_context}</p>
    <p style="font-size: 0.9em; color: #666;">Votre assistant intelligent pour des projets immobiliers réussis</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Help section
with st.expander("💡 Guide d'Utilisation Avancée"):
    st.markdown("""
    #### Comment utiliser l'assistant:
    
    **💰 Pour l'analyse budgétaire:**
    - Mentionnez votre budget: "J'ai 250000 DT" ou "250 mille dinars"
    - Précisez la ville: "pour un terrain à Sousse"
    - L'assistant mémorise vos préférences pour la suite
    
    **🔍 Pour la recherche de propriétés:**
    - Utilisez des mots comme "cherche", "trouve", "recommande", "montre"
    - Spécifiez le type: "villa", "appartement", "terrain", "maison"
    - L'assistant propose les meilleures correspondances avec URLs réelles
    
    **🧠 Mémoire conversationnelle:**
    - L'assistant se souvient de votre budget et ville préférée
    - Continuez la conversation naturellement
    - Changez vos critères à tout moment
    
    **📊 Fonctionnalités avancées:**
    - Analyse de budget intelligente avec recommandations personnalisées
    - Recherche de propriétés avec tri par pertinence  
    - Statistiques du marché en temps réel
    - URLs vers les annonces réelles
    - Descriptions détaillées et caractéristiques
    - Conseils d'investissement contextuels
    
    **💡 Exemples d'utilisation:**
    - "J'ai 300000 DT pour une villa à Tunis"
    - "Montre-moi des terrains à Sousse"
    - "Cherche des appartements avec mon budget"
    - "Recommande des propriétés dans ma gamme de prix"
    """)
