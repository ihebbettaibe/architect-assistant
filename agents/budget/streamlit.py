
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
    from plotly.subplots import make_subplots
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
            'user_profile': {'name': 'Ranim'},
            'search_history': [],
            'preferences': {},
            'last_budget': None,
            'last_city': None,
            'interaction_count': 0
        }
        # Load all CSVs from cleaned_data/
        self.sample_properties = self._load_properties_from_csv()

    def _load_properties_from_csv(self) -> List[Dict]:
        """Load all property data from cleaned_data/*.csv files"""
        import glob
        all_properties = []
        csv_dir = os.path.join(os.path.dirname(__file__), '../../cleaned_data')
        csv_files = glob.glob(os.path.join(csv_dir, 'cleaned_*_properties.csv'))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                # Standardize columns for downstream code
                for _, row in df.iterrows():
                    prop = {
                        'ID': row.get('ID') or row.get('id') or '',
                        'Title': row.get('Title') or row.get('title') or row.get('Titre') or '',
                        'Price': int(row.get('Price') or row.get('Prix') or row.get('price') or 0),
                        'Surface': int(row.get('Surface') or row.get('surface') or 0),
                        'Location': row.get('Location') or row.get('location') or '',
                        'URL': row.get('URL') or row.get('url') or '',
                        'Type': row.get('Type') or row.get('type') or '',
                        'City': row.get('City') or row.get('city') or '',
                        'Neighborhood': row.get('Neighborhood') or row.get('neighborhood') or '',
                        'Description': row.get('Description') or row.get('description') or '',
                        'Features': row.get('Features') or row.get('features') or [],
                        'Contact': row.get('Contact') or row.get('contact') or '',
                        'Posted_Date': row.get('Posted_Date') or row.get('posted_date') or '',
                        'Quality_Score': int(row.get('Quality_Score') or row.get('quality_score') or 3),
                        'Agent_Name': row.get('Agent_Name') or row.get('agent_name') or '',
                        'Views': int(row.get('Views') or row.get('views') or 0),
                        'Image_URL': row.get('Image_URL') or row.get('image_url') or '',
                    }
                    # If Features is a string, try to split it
                    if isinstance(prop['Features'], str):
                        prop['Features'] = [f.strip() for f in prop['Features'].split(',') if f.strip()]
                    all_properties.append(prop)
            except Exception as e:
                print(f"⚠️ Failed to load {csv_path}: {e}")
        return all_properties
    
    def _generate_property_description(self, prop_type: str, surface: int, neighborhood: str, city: str) -> str:
        """Generate realistic property descriptions"""
        descriptions = {
            'Terrain': [
                f"Excellent terrain de {surface}m² situé dans le quartier prisé de {neighborhood}. Idéal pour construction résidentielle.",
                f"Terrain constructible de {surface}m² avec toutes les commodités à proximité. Raccordé aux réseaux.",
                f"Belle opportunité d'investissement : terrain de {surface}m² bien exposé avec vue dégagée."
            ],
            'Villa': [
                f"Magnifique villa de {surface}m² avec jardin et piscine à {neighborhood}. Parfait pour famille nombreuse.",
                f"Villa moderne de {surface}m² entièrement rénovée, quartier calme et sécurisé.",
                f"Splendide villa de {surface}m² avec vue panoramique sur {city}. Finitions haut de gamme."
            ],
            'Appartement': [
                f"Bel appartement de {surface}m² au cœur de {neighborhood}. Proche de tous services.",
                f"Appartement moderne de {surface}m² avec balcon et parking. État impeccable.",
                f"Charmant appartement de {surface}m² proche de toutes commodités. Lumineux et bien agencé."
            ],
            'Maison': [
                f"Maison familiale de {surface}m² avec cour et garage à {neighborhood}. Parfaite pour investissement.",
                f"Belle maison de {surface}m² entièrement rénovée. Cachet authentique préservé.",
                f"Maison traditionnelle de {surface}m² dans un quartier résidentiel calme."
            ]
        }
        return np.random.choice(descriptions.get(prop_type, ["Propriété à vendre"]))
    
    def _generate_property_features(self, prop_type: str) -> List[str]:
        """Generate property features based on type"""
        common_features = ["Proche écoles", "Transport public", "Commerces à proximité"]
        
        type_features = {
            'Terrain': ["Constructible", "Raccordé aux réseaux", "Bien exposé", "Titre foncier"],
            'Villa': ["Piscine", "Jardin", "Garage", "Terrasse", "Sécurisé"],
            'Appartement': ["Ascenseur", "Balcon", "Parking", "Climatisation", "Concierge"],
            'Maison': ["Cour", "Garage", "Terrasse", "Cave", "Rénovée"]
        }
        
        features = common_features + type_features.get(prop_type, [])
        return np.random.choice(features, size=min(5, len(features)), replace=False).tolist()
    
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
    
    def search_properties(self, budget: int, city: str = None, property_type: str = None, max_results: int = 12) -> List[Dict]:
        """Enhanced property search with filtering and ranking"""
        filtered_properties = []
        
        for prop in self.sample_properties:
            # Filter by budget (±30% tolerance for more results)
            if budget * 0.7 <= prop['Price'] <= budget * 1.3:
                # Filter by city if specified
                city_val = str(prop.get('City', '') or '')
                type_val = str(prop.get('Type', '') or '')
                if city is None or (isinstance(city_val, str) and city_val.lower() == city.lower()):
                    # Filter by property type if specified
                    if property_type is None or (isinstance(type_val, str) and type_val.lower() == property_type.lower()):
                        # Calculate match score
                        price_diff = abs(prop['Price'] - budget) / budget
                        match_score = 1 - price_diff
                        # Bonus for exact city match
                        if city and isinstance(city_val, str) and city_val.lower() == city.lower():
                            match_score += 0.2
                        # Bonus for exact property type match
                        if property_type and isinstance(type_val, str) and type_val.lower() == property_type.lower():
                            match_score += 0.1
                        # Bonus for quality score
                        match_score += (prop['Quality_Score'] - 3) * 0.05
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
        
        # Quality-based recommendations
        if prop['Quality_Score'] >= 5:
            reasons.append("⭐ Propriété premium")
        elif prop['Quality_Score'] >= 4:
            reasons.append("✨ Très bonne qualité")
        
        # Surface-based recommendations
        surface_per_dt = prop['Surface'] / prop['Price']
        if surface_per_dt > 0.002:  # Good surface per DT ratio
            reasons.append("📐 Excellente surface pour le prix")
        
        # Location-based recommendations
        if city and prop['City'].lower() == city.lower():
            reasons.append("📍 Dans votre ville préférée")
        
        # Recent posting
        posted_date_str = prop.get('Posted_Date', '')
        # Only try to parse if it's a non-empty, non-NaN string
        if posted_date_str and isinstance(posted_date_str, str):
            try:
                # Some CSVs may have NaN as float, skip those
                if posted_date_str.lower() != 'nan':
                    posted_date = datetime.strptime(posted_date_str, '%Y-%m-%d')
                    days_ago = (datetime.now() - posted_date).days
                    if days_ago <= 7:
                        reasons.append("🆕 Annonce récente")
            except Exception:
                pass
        
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
                    "Analyser les tendances du marché",
                    "Afficher les statistiques détaillées"
                ])
            else:
                suggestions.extend([
                    "Rechercher des propriétés dans ce budget",
                    "Voir les options dans d'autres villes",
                    "Analyser le marché local",
                    "Comparer les prix par m²"
                ])
        else:
            suggestions.extend([
                "Préciser votre budget disponible",
                "Indiquer votre ville d'intérêt",
                "Spécifier le type de propriété souhaité",
                "Voir les tendances du marché"
            ])
        
        # Add memory-based suggestions
        if self.conversation_memory['interaction_count'] > 1:
            suggestions.append("Revoir nos échanges précédents")
        
        if self.conversation_memory['search_history']:
            suggestions.append("Comparer avec vos recherches précédentes")
        
        return suggestions[:6]  # Limit to 6 suggestions

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

# Enhanced Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .main-header h1 {
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-bottom: 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.1; }
        50% { transform: scale(1.1); opacity: 0.2; }
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: slideIn 0.4s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chat-user {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #232946;
        border-left: 5px solid #764ba2;
    }

    .chat-agent {
        background: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 100%);
        color: #232946;
        border-left: 5px solid #667eea;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .property-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 100%);
        padding: 0;
        border-radius: 20px;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
        border: 1px solid #c7d2fe;
        overflow: hidden;
    }
    
    .property-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .property-header {
        background: linear-gradient(135deg, #667eea 0%, #a5b4fc 100%);
        color: #232946;
        padding: 1.5rem;
        margin-bottom: 0;
    }
    
    .property-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .property-price {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffd700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .property-content {
        padding: 1.5rem;
        background: #e0e7ff;
        color: #232946;
    }
    
    .property-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 0;
        margin-bottom: 1rem;
    }
    
    .property-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .property-info-item {
        background: #c7d2fe;
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #232946;
    }
    
    .property-info-label {
        font-size: 0.85rem;
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .property-info-value {
        font-size: 1rem;
        color: #232946;
        font-weight: 700;
    }
    
    .property-description {
        background: #c7d2fe;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        font-style: italic;
        color: #232946;
        line-height: 1.6;
    }
    
    .property-features {
        margin: 1rem 0;
    }
    
    .feature-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 600;
    }
    
    .property-reasons {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #10b981;
        margin: 1rem 0;
        color: #232946;
    }
    
    .reasons-title {
        color: #047857;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .reason-item {
        color: #065f46;
        font-size: 0.85rem;
        margin: 0.2rem 0;
        line-height: 1.4;
    }
    
    .property-footer {
        background: #c7d2fe;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #a5b4fc;
        color: #232946;
    }
    
    .contact-info {
        color: #475569;
        font-weight: 600;
    }
    
    .view-link {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .view-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        text-decoration: none;
        color: white !important;
    }
    
    .quality-stars {
        color: #fbbf24;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .suggestion-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        border-radius: 20px;
        background: #e8f0fe;
        color: #333;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    
    .suggestion-chip:hover {
        background: #667eea;
        color: white;
    }
    
    .stButton>button {
        background: #667eea;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #764ba2;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = SimpleBudgetAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_suggestion' not in st.session_state:
    st.session_state.selected_suggestion = None

# Main application
def main():
    agent = st.session_state.agent
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Agent Budget Immobilier</h1>
        <p>Trouvez la propriété parfaite en Tunisie avec notre assistant intelligent</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar (minimal content)
    with st.sidebar:
        st.header("🤖 Agent Info")
        st.write("Agent Budget Immobilier actif")
        st.write(f"Utilisateur: {agent.conversation_memory['user_profile'].get('name', 'Mme Ranim')}")
        st.write(f"Interactions: {agent.conversation_memory['interaction_count']}")
        if agent.conversation_memory['last_budget']:
            st.write(f"Dernier budget: {agent.conversation_memory['last_budget']:,} DT")
        if agent.conversation_memory['last_city']:
            st.write(f"Dernière ville: {agent.conversation_memory['last_city']}")
    
    # Main content
    st.header("💬 Conversation")
    
    # Chat input
    user_input = st.text_input(
        "Posez votre question ou décrivez ce que vous cherchez",
        placeholder="Ex: Je cherche un terrain à Tunis avec un budget de 300000 DT"
    )
    
    if user_input:
        # Process user message
        response = agent.process_message(user_input)
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_input
        })
        st.session_state.chat_history.append({
            'role': 'agent',
            'message': response['agent_response']
        })
        
        # Update session state with latest response
        st.session_state.last_response = response
        
        # Clear input
        st.session_state.user_input = ""
    
    # Display chat history
    user_name = st.session_state.agent.conversation_memory['user_profile'].get('name', 'Vous')
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f'<div class="chat-message chat-user">👤 <strong>{user_name}:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message chat-agent">🤖 <strong>Agent:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
    
    # Results and Suggestions section - moved under conversation
    st.header("🏡 Résultats & Suggestions")
        
   
    # Display suggestions
    if 'last_response' in st.session_state and st.session_state.last_response.get('suggestions'):
        st.subheader("Suggestions")
        for suggestion in st.session_state.last_response['suggestions']:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                st.session_state.selected_suggestion = suggestion
                # Process suggestion as new input
                response = agent.process_message(suggestion)
                st.session_state.chat_history.append({
                    'role': 'user',
                    'message': suggestion
                })
                st.session_state.chat_history.append({
                    'role': 'agent',
                    'message': response['agent_response']
                })
                st.session_state.last_response = response
    
    # Display properties if search was triggered
    if 'last_response' in st.session_state and st.session_state.last_response.get('should_search'):
        budget = st.session_state.last_response['budget_analysis'].get('extracted_budget')
        city = st.session_state.last_response['budget_analysis'].get('city')
        prop_type = st.session_state.last_response['budget_analysis'].get('property_type')
        
        properties = agent.search_properties(budget, city, prop_type)
        
        if properties:
            st.subheader("Propriétés Correspondantes")
            for i, prop in enumerate(properties[:6]):  # Show up to 6 properties
                card_html = f'''
                <div class="property-card">
                    <div class="property-header">
                        <div class="property-title">{prop['Title']}</div>
                        <div class="property-price">{prop['Price']:,} DT</div>
                    </div>
                    <div class="property-content">
                        <img src="{prop['Image_URL']}" class="property-image" alt="Photo de la propriété">
                        <div class="property-info-grid">
                            <div class="property-info-item">
                                <div class="property-info-label">Type</div>
                                <div class="property-info-value">🏠 {prop['Type']}</div>
                            </div>
                            <div class="property-info-item">
                                <div class="property-info-label">Surface</div>
                                <div class="property-info-value">📏 {prop['Surface']} m²</div>
                            </div>
                            <div class="property-info-item">
                                <div class="property-info-label">Localisation</div>
                                <div class="property-info-value">📍 {prop['Neighborhood']}, {prop['City']}</div>
                            </div>
                            <div class="property-info-item">
                                <div class="property-info-label">Publié le</div>
                                <div class="property-info-value">📅 {prop['Posted_Date']}</div>
                            </div>
                        </div>
                        <div class="property-description">
                            <strong>Description:</strong> {prop['Description']}
                        </div>
                        <div class="quality-stars">
                            <strong>Qualité:</strong> {'★' * prop['Quality_Score']} ({prop['Quality_Score']}/5)
                        </div>
                        <div class="property-features">
                            <strong style="color: #374151; margin-bottom: 0.5rem; display: block;">Caractéristiques:</strong>
                            {''.join([f'<span class="feature-tag">{feature}</span>' for feature in prop['Features']])}
                        </div>
                        <div class="property-reasons">
                            <div class="reasons-title">🎯 Pourquoi cette propriété ?</div>
                            {'<br>'.join([f'<div class="reason-item">• {reason}</div>' for reason in prop['recommendation_reason'].split(' • ')])}
                        </div>
                    </div>
                    <div class="property-footer">
                        <div class="contact-info">
                            📞 <strong>{prop['Contact']}</strong><br>
                            👤 Agent: <strong>{prop['Agent_Name']}</strong><br>
                            👁️ Vues: <strong>{prop['Views']}</strong>
                        </div>
                        <a href="{prop['URL']}" target="_blank" class="view-link">
                            🔗 Voir l'annonce complète
                        </a>
                    </div>
                </div>
                '''
                st.markdown(card_html, unsafe_allow_html=True)
                # Add some spacing between properties
                if i < len(properties[:6]) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("Aucune propriété trouvée correspondant à vos critères.")
    
    # Display market analysis if available
    if 'last_response' in st.session_state and st.session_state.last_response.get('should_search'):
        analysis = agent.analyze_client_budget({
            'budget': budget,
            'city': city
        })

        st.subheader("📊 Analyse du Marché")
        st.write(f"Nombre de propriétés trouvées: {analysis['market_statistics']['inventory_count']}")
        if analysis['market_statistics']['inventory_count'] > 0:
            # Enhanced: handle None and add better formatting
            price_mean = analysis['market_statistics']['price_stats'].get('mean')
            surface_mean = analysis['market_statistics']['surface_stats'].get('mean')
            feasibility = analysis['market_statistics']['budget_feasibility'].get('feasibility_ratio')

            price_mean_str = f"{int(price_mean):,} DT" if price_mean is not None else "N/A"
            surface_mean_str = f"{int(surface_mean)} m²" if surface_mean is not None else "N/A"
            feasibility_str = f"{feasibility:.1%}" if feasibility is not None else "N/A"

            st.markdown(f"""
            <div class="stats-grid">
                <div class="stats-card">
                    <div class="stats-title">Prix moyen</div>
                    <div class="stats-value">{price_mean_str}</div>
                    <div class="stats-label">Prix moyen des propriétés trouvées</div>
                </div>
                <div class="stats-card">
                    <div class="stats-title">Surface moyenne</div>
                    <div class="stats-value">{surface_mean_str}</div>
                    <div class="stats-label">Surface moyenne des propriétés</div>
                </div>
            <div class="stats-card">
                <div class="stats-title">Faisabilité</div>
                <div class="stats-value">{feasibility_str}</div>
                <div class="stats-label">Propriétés dans votre budget</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Optional Plotly chart
        if PLOTLY_AVAILABLE and analysis['market_statistics']['inventory_count'] > 0:
            prices = [p['Price'] for p in analysis['comparable_properties'] if p.get('Price') is not None]
            if prices:
                fig = px.histogram(
                    x=prices,
                    nbins=20,
                    title="Distribution des Prix",
                    labels={'x': 'Prix (DT)', 'y': 'Nombre de propriétés'},
                    template='plotly_white'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()