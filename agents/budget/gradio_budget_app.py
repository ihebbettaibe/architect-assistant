import gradio as gr
import sys
import os
import re
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import json

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

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
        
        return result
    
    def search_properties(self, budget: int, city: str = None, property_type: str = None, max_results: int = 6) -> List[Dict]:
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
        
        return {
            'agent_response': response,
            'budget_analysis': budget_info,
            'should_search': should_search,
            'reliability_score': budget_info['confidence'],
            'conversation_context': self.conversation_memory
        }

# Initialize the agent
agent = SimpleBudgetAgent()

def format_property_card(prop: Dict) -> str:
    """Format a property as an HTML card for display"""
    def clean_val(val, default='N/A'):
        if val is None:
            return default
        sval = str(val).strip()
        if sval.lower() in ['nan', 'none', 'n/a', '']:
            return default
        return sval

    title = clean_val(prop.get('Title'), 'Propriété sans titre')
    price = f"{prop.get('Price', 0):,} DT"
    type_val = clean_val(prop.get('Type'))
    surface = prop.get('Surface')
    surface_str = f"{surface} m²" if surface and str(surface).lower() not in ['nan', 'none', 'n/a', ''] and float(surface) > 0 else "N/A"
    neighborhood = clean_val(prop.get('Neighborhood'), '')
    city_val = clean_val(prop.get('City'), '')
    location_str = ', '.join([v for v in [neighborhood, city_val] if v and v != 'N/A']) or 'N/A'
    description = prop.get('Description', 'Description non disponible')
    quality_score = prop.get('Quality_Score', 0)
    reasons = prop.get('recommendation_reason', '')
    url = clean_val(prop.get('URL'), '#')
    
    # Generate better description if missing
    if not description or str(description).strip().lower() in ['aucune description.', 'aucune', 'n/a', 'none', 'nan', '']:
        if type_val != 'N/A' and city_val != 'N/A' and surface and surface > 0:
            description = f"Belle propriété de type {type_val.lower()} de {surface}m² située à {city_val}. Idéale pour investissement ou habitation."
        else:
            description = "Propriété intéressante avec un bon potentiel. Contactez-nous pour une visite."
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 15px;
        margin: 10px 0;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <div style="
            background: linear-gradient(135deg, #1e40af 0%, #5b21b6 100%);
            color: white;
            padding: 15px;
        ">
            <h3 style="margin: 0 0 5px 0; font-size: 1.2rem;">{title}</h3>
            <div style="font-size: 1.4rem; font-weight: bold; color: #ffd700;">{price}</div>
        </div>
        <div style="
            padding: 15px;
            background: #1e293b;
            color: white;
        ">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                <div><strong>Type:</strong> 🏠 {type_val}</div>
                <div><strong>Surface:</strong> 📏 {surface_str}</div>
                <div><strong>Localisation:</strong> 📍 {location_str}</div>
                <div><strong>Qualité:</strong> {'★' * quality_score}{'☆' * (5-quality_score)} ({quality_score}/5)</div>
            </div>
            <div style="margin: 10px 0; font-style: italic; line-height: 1.4;">
                <strong>Description:</strong> {description}
            </div>
            {f'<div style="color: #a7f3d0; margin: 10px 0;"><strong>🎯 Pourquoi cette propriété ?</strong><br>{reasons}</div>' if reasons else ''}
            <div style="text-align: center; margin-top: 15px;">
                <a href="{url}" target="_blank" style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    text-decoration: none;
                    font-weight: 600;
                ">🔗 Voir l'annonce complète</a>
            </div>
        </div>
    </div>
    """
    return card_html

def chat_response(message, history):
    """Process user message and return response with property cards"""
    if not message.strip():
        return "Veuillez poser une question ou décrire ce que vous cherchez."
    
    # Process the message
    response = agent.process_message(message)
    
    # Start with the agent's text response
    output = response['agent_response']
    
    # If properties should be searched, add them
    if response.get('should_search'):
        budget = response['budget_analysis'].get('extracted_budget')
        city = response['budget_analysis'].get('city')
        prop_type = response['budget_analysis'].get('property_type')
        
        if budget:
            properties = agent.search_properties(budget, city, prop_type)
            
            if properties:
                output += "\n\n## 🏡 Propriétés Correspondantes\n\n"
                
                # Add property cards
                for prop in properties:
                    output += format_property_card(prop) + "\n"
                
                # Add market analysis
                output += f"\n\n### 📊 Analyse du Marché\n"
                output += f"**Nombre de propriétés trouvées:** {len(properties)}\n\n"
                
                if len(properties) > 0:
                    valid_prices = [p['Price'] for p in properties if p.get('Price', 0) > 0]
                    valid_surfaces = [p['Surface'] for p in properties if p.get('Surface', 0) > 0]
                    
                    if valid_prices:
                        price_mean = np.mean(valid_prices)
                        output += f"**Prix moyen:** {int(price_mean):,} DT\n"
                    
                    if valid_surfaces:
                        surface_mean = np.mean(valid_surfaces)
                        output += f"**Surface moyenne:** {int(surface_mean)} m²\n"
                    
                    feasibility = len([p for p in properties if p.get('Price', 0) <= budget]) / len(properties)
                    output += f"**Faisabilité:** {feasibility:.1%} des propriétés dans votre budget\n"
            else:
                output += "\n\n⚠️ Aucune propriété trouvée correspondant à vos critères."
    
    return output

def get_agent_info():
    """Return agent information for display"""
    info = f"""
    ## 🤖 Agent Budget Immobilier
    
    **Statut:** Actif ✅
    **Utilisateur:** {agent.conversation_memory['user_profile'].get('name', 'Mme Ranim')}
    **Interactions:** {agent.conversation_memory['interaction_count']}
    **Propriétés en base:** {len(agent.sample_properties):,}
    """
    
    if agent.conversation_memory['last_budget']:
        info += f"\n**Dernier budget:** {agent.conversation_memory['last_budget']:,} DT"
    
    if agent.conversation_memory['last_city']:
        info += f"\n**Dernière ville:** {agent.conversation_memory['last_city']}"
    
    return info

def clear_conversation():
    """Clear conversation history"""
    global agent
    agent.conversation_memory['interaction_count'] = 0
    agent.conversation_memory['search_history'] = []
    agent.conversation_memory['last_budget'] = None
    agent.conversation_memory['last_city'] = None
    return "Conversation effacée! 🧹", ""

# Custom CSS for better styling
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.property-card {
    margin: 10px 0;
}
"""

# Create the Gradio interface
with gr.Blocks(
    title="🏗️ Agent Budget Immobilier",
    css=css,
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🏗️ Agent Budget Immobilier
    ### Trouvez la propriété parfaite en Tunisie avec notre assistant intelligent
    
    Décrivez vos préférences de budget et de localisation, et je vous aiderai à trouver les meilleures propriétés disponibles.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                type="messages",
                height=500,
                show_label=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ex: Je cherche un terrain à Tunis avec un budget de 300000 DT",
                    show_label=False,
                    scale=4
                )
                submit = gr.Button("Envoyer", scale=1, variant="primary")
            
            with gr.Row():
                clear = gr.Button("🗑️ Effacer la conversation", variant="secondary")
                
        with gr.Column(scale=1):
            agent_info = gr.Markdown(get_agent_info())
            
            gr.Markdown("""
            ### 💡 Exemples de requêtes:
            - "Je veux construire une maison à Sousse avec un budget de 300k DT"
            - "Trouve-moi un terrain à Tunis pour 200000 dinars"
            - "Propriétés disponibles à Sfax dans mon budget"
            - "Recommande-moi quelque chose de bien"
            """)
            
            gr.Markdown("""
            ### 🎯 Fonctionnalités:
            - ✅ Recherche intelligente par budget
            - ✅ Filtrage par ville
            - ✅ Analyse du marché immobilier
            - ✅ Recommandations personnalisées
            - ✅ Mémoire conversationnelle
            """)

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        # Get bot response
        bot_message = chat_response(message, chat_history)
        
        # Add to chat history (new format)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return "", chat_history

    def clear_chat():
        clear_conversation()
        return [], get_agent_info()

    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, [], [chatbot, agent_info])
    
    # Update agent info when chat changes
    chatbot.change(lambda: get_agent_info(), [], agent_info)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
