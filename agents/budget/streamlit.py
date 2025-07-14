
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
        
        for i in range(150):  # Increased to 150 properties
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
            
            # Generate quality score based on features
            quality_score = np.random.randint(3, 6)  # 3-5 stars
            
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
                'Posted_Date': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d'),
                'Quality_Score': quality_score,
                'Agent_Name': f'{np.random.choice(["Ahmed", "Fatma", "Mohamed", "Salma", "Karim", "Amina"])} {np.random.choice(["Ben Ali", "Trabelsi", "Sassi", "Khedira", "Bouazizi"])}',
                'Views': np.random.randint(50, 500),
                'Image_URL': f'https://images.unsplash.com/photo-{np.random.randint(1500000000, 1600000000)}-{np.random.randint(100000, 999999)}?w=400&h=300&fit=crop'
            })
        
        return properties
    
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
                if city is None or prop['City'].lower() == city.lower():
                    # Filter by property type if specified
                    if property_type is None or prop['Type'].lower() == property_type.lower():
                        # Calculate match score
                        price_diff = abs(prop['Price'] - budget) / budget
                        match_score = 1 - price_diff
                        
                        # Bonus for exact city match
                        if city and prop['City'].lower() == city.lower():
                            match_score += 0.2
                        
                        # Bonus for exact property type match
                        if property_type and prop['Type'].lower() == property_type.lower():
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
        posted_date = datetime.strptime(prop['Posted_Date'], '%Y-%m-%d')
        days_ago = (datetime.now() - posted_date).days
        if days_ago <= 7:
            reasons.append("🆕 Annonce récente")
        
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
        background: linear-gradient(135deg, #667eea         0%, #764ba2 100%);
        color: white;
        border-left: 5px solid #764ba2;
    }
    
    .chat-agent {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
        border-left: 5px solid #667eea;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .property-card {
        background: #fefae0;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .property-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
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
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f'<div class="chat-message chat-user">👤 <strong>Vous:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
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
            for prop in properties[:6]:  # Show up to 6 properties
                with st.expander(f"{prop['Title']} - {prop['Price']:,} DT"):
                    st.markdown(f"""
                    <div class="property-card">
                        <img src="{prop['Image_URL']}" style="width:100%; border-radius:10px; margin-bottom:1rem;">
                        <p><strong>🏠 Type:</strong> {prop['Type']}</p>
                        <p><strong>📍 Lieu:</strong> {prop['Location']}</p>
                        <p><strong>📏 Surface:</strong> {prop['Surface']} m²</p>
                        <p><strong>💰 Prix:</strong> {prop['Price']:,} DT</p>
                        <p><strong>📝 Description:</strong> {prop['Description']}</p>
                        <p><strong>✨ Caractéristiques:</strong> {', '.join(prop['Features'])}</p>
                        <p><strong>📅 Publié:</strong> {prop['Posted_Date']}</p>
                        <p><strong>⭐ Qualité:</strong> {'★' * prop['Quality_Score']}</p>
                        <p><strong>🔍 Raison:</strong> {prop['recommendation_reason']}</p>
                        <p><strong>📞 Contact:</strong> {prop['Contact']}</p>
                        <p><a href="{prop['URL']}" target="_blank">Voir l'annonce</a></p>
                    </div>
                    """, unsafe_allow_html=True)
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
            st.write(f"Prix moyen: {int(analysis['market_statistics']['price_stats']['mean']):,} DT")
            st.write(f"Surface moyenne: {int(analysis['market_statistics']['surface_stats']['mean'])} m²")
            feasibility = analysis['market_statistics']['budget_feasibility']['feasibility_ratio']
            st.write(f"Ratio de faisabilité: {feasibility:.1%}")
            
            # Optional Plotly chart
            if PLOTLY_AVAILABLE and analysis['market_statistics']['inventory_count'] > 0:
                prices = [p['Price'] for p in analysis['comparable_properties']]
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