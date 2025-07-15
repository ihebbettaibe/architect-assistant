import sys
import os
import re
from datetime import datetime
from typing import Dict, List, Any

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

import streamlit as st
import pandas as pd
import numpy as np

# Try to import optional dependencies
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Enhanced Budget Agent Implementation
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
            
            # Get the correct path to cleaned_data
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_dir = os.path.abspath(os.path.join(current_dir, '../../cleaned_data'))
            
            print(f"🔍 Looking for CSV files in: {csv_dir}")
            
            # Check if directory exists
            if not os.path.exists(csv_dir):
                print(f"❌ Directory does not exist: {csv_dir}")
                return all_properties
            
            csv_files = glob.glob(os.path.join(csv_dir, 'cleaned_*_properties.csv'))
            print(f"📂 Found {len(csv_files)} CSV files")
            
            for csv_path in csv_files:
                try:
                    print(f"📄 Loading file: {csv_path}")
                    df = pd.read_csv(csv_path)
                    print(f"✅ Loaded {len(df)} rows from {os.path.basename(csv_path)}")
                    
                    # Standardize columns for downstream code
                    for _, row in df.iterrows():
                        prop = {
                            'ID': row.get('ID') or row.get('id') or '',
                            'Title': row.get('Title') or row.get('title') or row.get('Titre') or '',
                            'Price': int(float(row.get('Price') or row.get('Prix') or row.get('price') or 0)),
                            'Surface': int(float(row.get('Surface') or row.get('surface') or 0)),
                            'Location': row.get('Location') or row.get('location') or '',
                            'URL': row.get('URL') or row.get('url') or '',
                            'Type': row.get('Type') or row.get('type') or '',
                            'City': row.get('City') or row.get('city') or '',
                            'Neighborhood': row.get('Neighborhood') or row.get('neighborhood') or '',
                            'Description': row.get('Description') or row.get('description') or '',
                            'Features': row.get('Features') or row.get('features') or [],
                            'Contact': row.get('Contact') or row.get('contact') or '',
                            'Posted_Date': row.get('Posted_Date') or row.get('posted_date') or '',
                            'Quality_Score': int(float(row.get('Quality_Score') or row.get('quality_score') or 3)),
                            'Agent_Name': row.get('Agent_Name') or row.get('agent_name') or '',
                            'Views': int(float(row.get('Views') or row.get('views') or 0)),
                            'Image_URL': row.get('Image_URL') or row.get('image_url') or '',
                        }
                        # If Features is a string, try to split it
                        if isinstance(prop['Features'], str):
                            prop['Features'] = [f.strip() for f in prop['Features'].split(',') if f.strip()]
                        all_properties.append(prop)
                except Exception as e:
                    print(f"⚠️ Failed to load {csv_path}: {str(e)}")
                    continue
            
            print(f"🏠 Total properties loaded: {len(all_properties)}")
            return all_properties
    
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
        """Property search: ignore property type, only filter by budget and city"""
        filtered_properties = []
        for prop in self.sample_properties:
            # Filter by budget (±30% tolerance for more results)
            if budget * 0.7 <= prop['Price'] <= budget * 1.3:
                city_val = str(prop.get('City', '') or '')
                city_match = (
                    city is None or
                    (isinstance(city_val, str) and city and city.lower() in city_val.lower())
                )
                if city_match:
                    # Calculate match score
                    price_diff = abs(prop['Price'] - budget) / budget
                    match_score = 1 - price_diff
                    # Bonus for city substring match
                    if city and isinstance(city_val, str) and city.lower() in city_val.lower():
                        match_score += 0.15
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
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 0;
        border-radius: 20px;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid #4f46e5;
        overflow: hidden;
    }
    
    .property-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .property-header {
        background: linear-gradient(135deg, #1e40af 0%, #5b21b6 100%);
        color: white;
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
        background: #1e293b;
        color: white;
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
        background: #334155;
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 4px solid #60a5fa;
        color: white;
    }
    
    .property-info-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .property-info-value {
        font-size: 1rem;
        color: white;
        font-weight: 700;
    }
    
    .property-description {
        background: #334155;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        font-style: italic;
        color: white;
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
        background: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #10b981;
        margin: 1rem 0;
        color: white;
    }
    
    .reasons-title {
        color: #22d3ee;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .reason-item {
        color: #a7f3d0;
        font-size: 0.85rem;
        margin: 0.2rem 0;
        line-height: 1.4;
    }
    
    .property-footer {
        background: #334155;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #475569;
        color: white;
    }
    
    .contact-info {
        color: #cbd5e1;
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
    
    /* New stats cards styling */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e0e7ff;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .stats-title {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .stats-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4338ca;
        margin: 0.5rem 0;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }
        
        .property-info-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = SimpleBudgetAgent()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'show_orchestrator' not in st.session_state:
        st.session_state.show_orchestrator = False
    if 'next_agent' not in st.session_state:
        st.session_state.next_agent = None
    if 'financial_analysis' not in st.session_state:
        st.session_state.financial_analysis = False
    
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

        # Data info
        st.markdown("---")
        st.subheader("Données")
        st.write(f"Propriétés: {len(agent.sample_properties)}")
    
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
    
    # Results section - moved under conversation
    st.header("🏡 Résultats")
    
    # Display properties if search was triggered
    if 'last_response' in st.session_state and st.session_state.last_response.get('should_search'):
        budget = st.session_state.last_response['budget_analysis'].get('extracted_budget')
        city = st.session_state.last_response['budget_analysis'].get('city')
        prop_type = st.session_state.last_response['budget_analysis'].get('property_type')

        # Get properties ONCE and use for both display and analysis
        properties = agent.search_properties(budget, city, prop_type)

        if properties:
            st.subheader("Propriétés Correspondantes")
            for i, prop in enumerate(properties[:6]):  # Show up to 6 properties
                # Clean up missing/empty fields
                def clean_val(val, default='N/A'):
                    if val is None:
                        return default
                    sval = str(val).strip()
                    if sval.lower() in ['nan', 'none', 'n/a', '']:
                        return default
                    return sval

                title = clean_val(prop.get('Title'), 'Propriété sans titre')
                price = f"{prop.get('Price', 0):,} DT"
                image_url = clean_val(prop.get('Image_URL'), '')
                if not image_url:
                    image_url = 'https://via.placeholder.com/400x200?text=Pas+de+photo'
                type_val = clean_val(prop.get('Type'))
                surface = prop.get('Surface')
                surface_str = f"{surface} m²" if surface and str(surface).lower() not in ['nan', 'none', 'n/a', ''] and float(surface) > 0 else "N/A"
                neighborhood = clean_val(prop.get('Neighborhood'), '')
                city_val = clean_val(prop.get('City'), '')
                location_str = ', '.join([v for v in [neighborhood, city_val] if v and v != 'N/A']) or 'N/A'
                posted_date = clean_val(prop.get('Posted_Date'), None)
                description = prop.get('Description')
                
                # Generate description if missing
                if not description or str(description).strip().lower() in ['aucune description.', 'aucune', 'n/a', 'none', 'nan', '']:
                    if type_val != 'N/A' and city_val != 'N/A' and surface and surface > 0:
                        description = f"Belle propriété de type {type_val.lower()} de {surface}m² située à {city_val}. Idéale pour investissement ou habitation."
                    elif type_val != 'N/A' and city_val != 'N/A':
                        description = f"Propriété de type {type_val.lower()} située à {city_val}. Excellent emplacement et bon potentiel."
                    elif city_val != 'N/A' and surface and surface > 0:
                        description = f"Propriété de {surface}m² située à {city_val}. Opportunité à saisir dans cette zone recherchée."
                    elif type_val != 'N/A':
                        description = f"Belle propriété de type {type_val.lower()}. Contactez-nous pour plus d'informations."
                    else:
                        description = "Propriété intéressante avec un bon potentiel. Contactez-nous pour une visite."
                
                quality_score = prop.get('Quality_Score', 0)
                if not isinstance(quality_score, int):
                    try:
                        quality_score = int(float(quality_score))
                        quality_score = max(0, min(5, quality_score))  # Clamp between 0-5
                    except (ValueError, TypeError):
                        quality_score = 0
                
                features = prop.get('Features')
                if not features or (isinstance(features, str) and features.strip().lower() in ['aucune', 'n/a', 'none', 'nan', '']):
                    # Generate better features based on property type
                    if type_val.lower() in ['terrain', 'lot']:
                        features = ["Constructible", "Raccordé aux réseaux", "Bien exposé", "Proche commodités"]
                    elif type_val.lower() in ['villa', 'maison']:
                        features = ["Jardin", "Garage", "Terrasse", "Quartier calme"]
                    elif type_val.lower() == 'appartement':
                        features = ["Balcon", "Parking", "Ascenseur", "Proche transports"]
                    else:
                        features = ["Bon emplacement", "Proche écoles", "Transport public", "Commerces à proximité"]
                elif isinstance(features, str):
                    features = [f.strip() for f in features.split(',') if f.strip()]
                
                # Ensure we have at least 3-4 good features
                if len(features) < 3:
                    extra_features = ["Proche écoles", "Transport public", "Commerces à proximité", "Quartier résidentiel"]
                    for feat in extra_features:
                        if feat not in features and len(features) < 4:
                            features.append(feat)
                
                contact = clean_val(prop.get('Contact'), None)
                agent_name = clean_val(prop.get('Agent_Name'), None)
                views = clean_val(prop.get('Views'), None)
                url = clean_val(prop.get('URL'), '#')
                reasons = prop.get('recommendation_reason') or ''
                
                reasons_html = ''
                if reasons:
                    reasons_list = [reason.strip() for reason in reasons.split(' • ') if reason.strip()]
                    reasons_html = '<br>'.join([f'<div class="reason-item">• {reason}</div>' for reason in reasons_list])
                
                # Only show posted_date if it is not missing or N/A
                posted_date_html = ''
                if posted_date and posted_date != 'N/A':
                    posted_date_html = f"<div class=\"property-info-item\"><div class=\"property-info-label\">Publié le</div><div class=\"property-info-value\">📅 {posted_date}</div></div>"
                
                # Only show contact/agent/views if not empty/default
                contact_html = ''
                if contact and contact != 'N/A':
                    contact_html += f'📞 <strong>{contact}</strong><br>'
                if agent_name and agent_name != 'N/A':
                    contact_html += f'👤 Agent: <strong>{agent_name}</strong><br>'
                if views and views != '0' and views != 'N/A':
                    contact_html += f'👁️ Vues: <strong>{views}</strong>'
                
                # Build footer content
                footer_content = ''
                if contact_html.strip():
                    footer_content = f'<div class="contact-info">{contact_html}</div>'
                footer_content += f'<a href="{url}" target="_blank" class="view-link">🔗 Voir l\'annonce complète</a>'
                
                card_html = f"<div class=\"property-card\"><div class=\"property-header\"><div class=\"property-title\">{title}</div><div class=\"property-price\">{price}</div></div><div class=\"property-content\"><div class=\"property-info-grid\"><div class=\"property-info-item\"><div class=\"property-info-label\">Type</div><div class=\"property-info-value\">🏠 {type_val}</div></div><div class=\"property-info-item\"><div class=\"property-info-label\">Surface</div><div class=\"property-info-value\">📏 {surface_str}</div></div><div class=\"property-info-item\"><div class=\"property-info-label\">Localisation</div><div class=\"property-info-value\">📍 {location_str}</div></div>{posted_date_html}</div><div class=\"property-description\"><strong>Description:</strong> {description}</div><div class=\"quality-stars\"><strong>Qualité:</strong> {'★' * quality_score}{'☆' * (5-quality_score)} ({quality_score}/5)</div><div class=\"property-features\"><strong style=\"color: #94a3b8; margin-bottom: 0.5rem; display: block;\">Caractéristiques:</strong>{''.join([f'<span class=\"feature-tag\">{feature}</span>' for feature in features])}</div><div class=\"property-reasons\"><div class=\"reasons-title\">🎯 Pourquoi cette propriété ?</div>{reasons_html if reasons_html else '<div class=\"reason-item\">Aucune raison spécifique.</div>'}</div></div><div class=\"property-footer\">{footer_content}</div></div>"
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add selection button for each property
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"✅ Choisir cette propriété", key=f"select_property_{i}"):
                        st.session_state.selected_property = prop
                        st.session_state.show_orchestrator = True
                        st.success(f"🎯 Propriété sélectionnée: {title}")
                        st.rerun()
                
                # Add some spacing between properties
                if i < len(properties[:6]) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)

            # Market analysis for displayed properties
            st.subheader("📊 Analyse du Marché")
            st.write(f"Nombre de propriétés trouvées: {len(properties)}")
            
            if len(properties) > 0:
                # Calculate statistics with proper error handling
                valid_prices = [p['Price'] for p in properties if p.get('Price') is not None and isinstance(p['Price'], (int, float)) and p['Price'] > 0]
                valid_surfaces = [p['Surface'] for p in properties if p.get('Surface') is not None and isinstance(p['Surface'], (int, float)) and p['Surface'] > 0]
                
                price_mean = np.mean(valid_prices) if valid_prices else None
                surface_mean = np.mean(valid_surfaces) if valid_surfaces else None
                feasibility = len([p for p in properties if p.get('Price') and p['Price'] <= budget]) / len(properties) if len(properties) > 0 and budget else 0

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
                if PLOTLY_AVAILABLE and valid_prices:
                    fig = px.histogram(
                        x=valid_prices,
                        nbins=min(20, len(valid_prices)),  # Adjust bins based on data size
                        title="Distribution des Prix",
                        labels={'x': 'Prix (DT)', 'y': 'Nombre de propriétés'},
                        template='plotly_white'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune propriété trouvée correspondant à vos critères.")
    
    # Orchestrator Agent Section
    if st.session_state.get('show_orchestrator', False) and st.session_state.get('selected_property'):
        st.markdown("---")
        st.header("🤖 Agent Orchestrateur - Prochaines Étapes")
        
        selected_prop = st.session_state.selected_property
        
        st.success(f"✅ **Propriété choisie:** {selected_prop.get('Title', 'Propriété sélectionnée')}")
        st.info(f"💰 **Prix:** {selected_prop.get('Price', 0):,} DT | 📏 **Surface:** {selected_prop.get('Surface', 'N/A')} m²")
        
        # Orchestrator decision logic
        st.subheader("🎯 Plan d'Action Recommandé")
        
        # Analyze the selected property
        property_price = selected_prop.get('Price', 0)
        property_type = selected_prop.get('Type', '')
        property_city = selected_prop.get('City', '')
        
        action_plan = []
        
        # Generate action plan based on property characteristics
        if property_type.lower() in ['terrain', 'lot']:
            action_plan.extend([
                "🏗️ **Étape 1:** Consultation avec l'Agent Design pour la planification architecturale",
                "📋 **Étape 2:** Vérification des autorisations de construction",
                "💰 **Étape 3:** Analyse détaillée du budget de construction",
                "📐 **Étape 4:** Conception des plans préliminaires"
            ])
        else:
            action_plan.extend([
                "🔍 **Étape 1:** Inspection détaillée de la propriété",
                "💰 **Étape 2:** Négociation du prix avec le vendeur",
                "📋 **Étape 3:** Vérification juridique et administrative",
                "🏠 **Étape 4:** Planification d'éventuelles rénovations"
            ])
        
        # Budget analysis
        user_budget = st.session_state.get('last_response', {}).get('budget_analysis', {}).get('extracted_budget', 0)
        if user_budget and property_price:
            remaining_budget = user_budget - property_price
            if remaining_budget > 0:
                action_plan.append(f"💵 **Budget restant:** {remaining_budget:,} DT pour les développements")
            else:
                action_plan.append("⚠️ **Attention:** Budget dépassé, négociation recommandée")
        
        for step in action_plan:
            st.markdown(f"- {step}")
        
        # Action buttons
        st.subheader("🚀 Actions Disponibles")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏗️ Agent Design", help="Consulter l'agent design pour la planification"):
                st.session_state.next_agent = "design"
                st.info("🔄 Redirection vers l'Agent Design...")
                st.info("💡 **Conseil:** L'Agent Design vous aidera à planifier l'architecture et la construction.")
        
        with col2:
            if st.button("📋 Agent Régulation", help="Vérifier les autorisations légales"):
                st.session_state.next_agent = "regulation"
                st.info("🔄 Redirection vers l'Agent Régulation...")
                st.info("💡 **Conseil:** L'Agent Régulation vérifiera toutes les conformités légales.")
        
        with col3:
            if st.button("💰 Analyse Financière", help="Analyse financière détaillée"):
                st.session_state.financial_analysis = True
                st.info("📊 Lancement de l'analyse financière détaillée...")
        
        with col4:
            if st.button("🔄 Nouvelle Recherche", help="Recommencer la recherche"):
                # Reset session state
                for key in ['selected_property', 'show_orchestrator', 'next_agent', 'financial_analysis']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Financial Analysis Section
        if st.session_state.get('financial_analysis', False):
            st.markdown("---")
            st.subheader("📊 Analyse Financière Détaillée")
            
            # Create financial breakdown
            property_cost = selected_prop.get('Price', 0)
            estimated_notary_fees = property_cost * 0.07  # 7% notaire
            estimated_agency_fees = property_cost * 0.03   # 3% agence
            estimated_taxes = property_cost * 0.02         # 2% taxes
            
            total_acquisition_cost = property_cost + estimated_notary_fees + estimated_agency_fees + estimated_taxes
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
                <h4 style="color: #60a5fa; margin-bottom: 1rem;">💰 Coûts d'Acquisition</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Prix de la propriété:</strong><br>
                        <span style="font-size: 1.2rem; color: #fbbf24;">{property_cost:,.0f} DT</span>
                    </div>
                    <div>
                        <strong>Frais de notaire (7%):</strong><br>
                        <span style="color: #f87171;">{estimated_notary_fees:,.0f} DT</span>
                    </div>
                    <div>
                        <strong>Frais d'agence (3%):</strong><br>
                        <span style="color: #f87171;">{estimated_agency_fees:,.0f} DT</span>
                    </div>
                    <div>
                        <strong>Taxes et frais (2%):</strong><br>
                        <span style="color: #f87171;">{estimated_taxes:,.0f} DT</span>
                    </div>
                </div>
                <hr style="margin: 1rem 0; border-color: #475569;">
                <div style="text-align: center;">
                    <strong style="font-size: 1.3rem;">Coût Total: <span style="color: #22d3ee;">{total_acquisition_cost:,.0f} DT</span></strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if user_budget:
                remaining_after_all_costs = user_budget - total_acquisition_cost
                if remaining_after_all_costs >= 0:
                    st.success(f"✅ Budget suffisant! Il vous restera {remaining_after_all_costs:,.0f} DT")
                else:
                    st.error(f"❌ Budget insuffisant! Il manque {abs(remaining_after_all_costs):,.0f} DT")
        
        # Next Agent Navigation
        if st.session_state.get('next_agent'):
            st.markdown("---")
            next_agent = st.session_state.next_agent
            
            if next_agent == "design":
                st.info("🏗️ **Prochaine étape:** Agent Design")
                st.markdown("""
                **L'Agent Design vous aidera avec:**
                - Conception architecturale
                - Plans de construction
                - Optimisation de l'espace
                - Recommandations de matériaux
                """)
                
                if st.button("▶️ Lancer l'Agent Design"):
                    st.info("🔄 Redirection vers l'application Agent Design...")
                    st.code("streamlit run agents/design_agent.py", language="bash")
            
            elif next_agent == "regulation":
                st.info("📋 **Prochaine étape:** Agent Régulation")
                st.markdown("""
                **L'Agent Régulation vous aidera avec:**
                - Vérification des autorisations
                - Conformité légale
                - Permis de construire
                - Régulations locales
                """)
                
                if st.button("▶️ Lancer l'Agent Régulation"):
                    st.info("🔄 Redirection vers l'application Agent Régulation...")
                    st.code("streamlit run agents/Regulation_agent.py", language="bash")

if __name__ == "__main__":
    main()