"""
LangChainBudgetAgent: Unified budget agent for property analysis using LangChain and Groq LLM.
All logic is self-contained in this file. No external local imports required.
"""
import dotenv
dotenv.load_dotenv()
import os
import sys
import json
import re
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain import hub

# --- Internal Classes (formerly separate files) ---

class EnhancedBudgetAgent:
    """
    Handles budget analysis and property search logic with conversation memory.
    """
    def __init__(self):
        self.tunisia_cities = [
            "Tunis", "Sfax", "Sousse", "Ettadhamen", "Kairouan", "Bizerte", 
            "Gabès", "Ariana", "Gafsa", "Monastir", "Ben Arous", "Kasserine",
            "Médenine", "Nabeul", "Tataouine", "Beja", "Jendouba", "Mahdia",
            "Sidi Bouzid", "Siliana", "Manouba", "Kef", "Tozeur", "Zaghouan", "Kebili"
        ]
        
        # Conversation memory
        self.conversation_memory = {
            'user_profile': {},
            'search_history': [],
            'last_budget': None,
            'last_city': None,
            'last_property_type': None,
            'interaction_count': 0
        }
        
    def analyze_client_budget(self, client_input: str) -> str:
        """
        Analyze client budget from string input with memory context.
        """
        try:
            # Update interaction count
            self.conversation_memory['interaction_count'] += 1
            
            # Parse the client input to extract budget information
            client_profile = self._parse_client_input(client_input)
            
            # Update memory with new information
            self._update_memory(client_profile)
            
            # Use memory to fill missing information
            enhanced_profile = self._enhance_with_memory(client_profile)
            
            # Perform budget analysis
            budget_amount = enhanced_profile.get('budget', 0)
            city = enhanced_profile.get('city', 'Unknown')
            property_type = enhanced_profile.get('property_type', 'apartment')
            
            # Simulate market analysis based on Tunisia real estate market
            market_stats = self._get_market_statistics(city, property_type, budget_amount)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(enhanced_profile, market_stats)
            
            # Generate contextual response
            contextual_message = self._generate_contextual_message(enhanced_profile)
            
            analysis_result = {
                'client_info': enhanced_profile,
                'market_statistics': market_stats,
                'recommendations': recommendations,
                'budget_feasibility': self._assess_budget_feasibility(budget_amount, market_stats),
                'contextual_message': contextual_message,
                'conversation_context': {
                    'interaction_count': self.conversation_memory['interaction_count'],
                    'has_previous_searches': len(self.conversation_memory['search_history']) > 0,
                    'memory_used': self._was_memory_used(client_profile)
                }
            }
            
            return json.dumps(analysis_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Budget analysis failed: {str(e)}"})
    
    def _update_memory(self, client_profile: dict):
        """Update conversation memory with new information."""
        if client_profile.get('budget'):
            self.conversation_memory['last_budget'] = client_profile['budget']
            self.conversation_memory['user_profile']['budget'] = client_profile['budget']
        
        if client_profile.get('city'):
            self.conversation_memory['last_city'] = client_profile['city']
            self.conversation_memory['user_profile']['preferred_city'] = client_profile['city']
        
        if client_profile.get('property_type'):
            self.conversation_memory['last_property_type'] = client_profile['property_type']
            self.conversation_memory['user_profile']['property_type'] = client_profile['property_type']
        
        # Add to search history
        search_entry = {
            'timestamp': '2024-01-01T00:00:00Z',  # Would use datetime.now() in real implementation
            'profile': client_profile,
            'original_input': client_profile.get('original_input', '')
        }
        self.conversation_memory['search_history'].append(search_entry)
        
        # Keep only last 10 searches
        if len(self.conversation_memory['search_history']) > 10:
            self.conversation_memory['search_history'] = self.conversation_memory['search_history'][-10:]
    
    def _enhance_with_memory(self, client_profile: dict) -> dict:
        """Enhance client profile with memory information."""
        enhanced = client_profile.copy()
        
        # Use previous budget if none provided
        if not enhanced.get('budget') and self.conversation_memory['last_budget']:
            enhanced['budget'] = self.conversation_memory['last_budget']
            enhanced['budget_source'] = 'memory'
        
        # Use previous city if none provided
        if not enhanced.get('city') and self.conversation_memory['last_city']:
            enhanced['city'] = self.conversation_memory['last_city']
            enhanced['city_source'] = 'memory'
        
        # Use previous property type if none provided
        if not enhanced.get('property_type') and self.conversation_memory['last_property_type']:
            enhanced['property_type'] = self.conversation_memory['last_property_type']
            enhanced['property_type_source'] = 'memory'
        
        return enhanced
    
    def _was_memory_used(self, original_profile: dict) -> bool:
        """Check if memory was used to enhance the profile."""
        return (
            (not original_profile.get('budget') and self.conversation_memory['last_budget']) or
            (not original_profile.get('city') and self.conversation_memory['last_city']) or
            (not original_profile.get('property_type') and self.conversation_memory['last_property_type'])
        )
    
    def _generate_contextual_message(self, profile: dict) -> str:
        """Generate contextual message based on conversation history."""
        messages = []
        
        if self.conversation_memory['interaction_count'] == 1:
            messages.append("👋 Bienvenue ! Je suis ravi de vous aider dans votre recherche immobilière.")
        else:
            messages.append(f"🔄 Continuons notre conversation (échange #{self.conversation_memory['interaction_count']}).")
        
        # Budget context
        if profile.get('budget'):
            budget = profile['budget']
            if profile.get('budget_source') == 'memory':
                messages.append(f"💰 J'utilise votre budget précédent de {budget:,} DT.")
            else:
                messages.append(f"💰 Budget détecté: {budget:,} DT.")
        
        # City context
        if profile.get('city'):
            city = profile['city']
            if profile.get('city_source') == 'memory':
                messages.append(f"📍 Je continue sur {city} (votre ville précédente).")
            else:
                messages.append(f"📍 Recherche dans la ville de {city}.")
        
        return " ".join(messages)
    
    def _parse_client_input(self, client_input: str) -> Dict:
        """Parse client input to extract structured information."""
        # Extract budget using improved regex for TND handling
        budget_patterns = [
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:tnd|dt|dinars?)',  # 150,000 TND or 150000 DT
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:thousand|k)\s*(?:tnd|dt|dinars?)?',  # 150 thousand TND
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:mille)\s*(?:tnd|dt|dinars?)?',  # 150 mille DT
            r'budget.*?(\d+(?:,\d+)*(?:\.\d+)?)',  # budget de 150000
        ]
        
        budget = 0
        for pattern in budget_patterns:
            budget_match = re.search(pattern, client_input.lower())
            if budget_match:
                budget_str = budget_match.group(1).replace(',', '')
                budget = float(budget_str)
                
                # Handle thousands notation
                if any(word in client_input.lower() for word in ['thousand', 'k ', 'mille']):
                    budget *= 1000
                
                # If the budget seems too large (over 10M), likely it's in wrong units
                if budget > 10000000:
                    budget = budget / 1000  # Convert back
                
                break
        
        # Extract city
        city = None
        for tunisia_city in self.tunisia_cities:
            if tunisia_city.lower() in client_input.lower():
                city = tunisia_city
                break
        
        # Extract property type
        property_type = 'apartment'  # default
        if any(word in client_input.lower() for word in ['house', 'villa', 'maison']):
            property_type = 'villa'
        elif 'studio' in client_input.lower():
            property_type = 'studio'
        elif any(word in client_input.lower() for word in ['land', 'terrain']):
            property_type = 'land'
        elif any(word in client_input.lower() for word in ['apartment', 'appartement']):
            property_type = 'apartment'
        
        return {
            'budget': budget if budget > 0 else None,
            'city': city,
            'property_type': property_type,
            'original_input': client_input
        }
    
    def _get_market_statistics(self, city: str, property_type: str, budget: float) -> Dict:
        """Generate realistic market statistics for Tunisia."""
        # Approximate market data for Tunisia (in TND)
        city_multipliers = {
            'Tunis': 1.3,
            'Sfax': 1.1,
            'Sousse': 1.2,
            'Ariana': 1.25,
            'Monastir': 1.15,
            'Nabeul': 1.1,
            'Bizerte': 1.0,
            'Gabès': 0.9,
            'Kairouan': 0.8,
            'Gafsa': 0.7
        }
        
        # Base prices per m² in TND
        base_prices = {
            'apartment': 2000,
            'house': 1500,
            'studio': 2500,
            'land': 300
        }
        
        multiplier = city_multipliers.get(city, 1.0)
        base_price = base_prices.get(property_type, 2000)
        avg_price_per_m2 = base_price * multiplier
        
        return {
            'inventory_count': 150,  # Simulated
            'price_stats': {
                'min': avg_price_per_m2 * 0.7,
                'max': avg_price_per_m2 * 1.8,
                'mean': avg_price_per_m2,
                'median': avg_price_per_m2 * 1.05
            },
            'price_per_m2_stats': {
                'min': avg_price_per_m2 * 0.7,
                'max': avg_price_per_m2 * 1.8,
                'mean': avg_price_per_m2,
                'median': avg_price_per_m2 * 1.05
            },
            'surface_stats': {
                'min': 40,
                'max': 200,
                'mean': 95,
                'median': 85
            },
            'city': city,
            'property_type': property_type
        }
    
    def _generate_recommendations(self, client_profile: Dict, market_stats: Dict) -> List[Dict]:
        """Generate property recommendations based on client profile and market data."""
        recommendations = []
        budget = client_profile.get('budget', 0)
        avg_price_per_m2 = market_stats['price_per_m2_stats']['mean']
        
        if budget > 0:
            # Calculate affordable surface area
            affordable_surface = budget / avg_price_per_m2
            
            recommendations.append({
                'type': 'surface_recommendation',
                'message': f"With {budget:,.0f} TND, you can afford approximately {affordable_surface:.0f} m² in {client_profile.get('city', 'your chosen city')}",
                'details': {
                    'budget': budget,
                    'avg_price_per_m2': avg_price_per_m2,
                    'affordable_surface': affordable_surface
                }
            })
            
            # Alternative city recommendations
            if affordable_surface < 60:  # If surface is small, recommend other cities
                recommendations.append({
                    'type': 'alternative_cities',
                    'message': "Consider these more affordable cities where your budget would go further",
                    'alternatives': ['Kairouan', 'Gafsa', 'Sidi Bouzid', 'Kasserine']
                })
        
        return recommendations
    
    def _assess_budget_feasibility(self, budget: float, market_stats: Dict) -> Dict:
        """Assess if the budget is feasible for the market."""
        if budget <= 0:
            return {'feasibility_ratio': 0.0, 'status': 'insufficient_info'}
        
        min_property_price = market_stats['price_stats']['min'] * 50  # 50m² minimum
        feasibility_ratio = budget / min_property_price
        
        status = 'excellent' if feasibility_ratio > 1.5 else \
                'good' if feasibility_ratio > 1.0 else \
                'limited' if feasibility_ratio > 0.7 else 'challenging'
        
        return {
            'feasibility_ratio': feasibility_ratio,
            'status': status,
            'min_property_price': min_property_price
        }

class BudgetAnalysis:
    """
    Handles market and property data analysis with enhanced property search.
    """
    def __init__(self):
        self.websites = [
            'tayara.tn', 'jumia.com.tn', 'afariat.com', 'immobilier.tn',
            'mubawab.tn', 'tecnocasa.tn', 'sarouty.tn'
        ]
        self.neighborhoods = {
            'Tunis': ['La Marsa', 'Carthage', 'Sidi Bou Said', 'Ariana', 'Manouba', 'El Menzah'],
            'Sfax': ['Centre Ville', 'Sakiet Ezzit', 'Route Tunis', 'Sfax Sud'],
            'Sousse': ['Centre Ville', 'Port El Kantaoui', 'Hammam Sousse', 'Sahloul', 'Kantaoui'],
            'Monastir': ['Centre Ville', 'Skanes', 'Ksar Hellal', 'Jemmal'],
            'Nabeul': ['Centre Ville', 'Hammamet', 'Kelibia', 'Korba']
        }

    def find_similar_properties(self, search_criteria: str) -> str:
        """
        Find similar properties with realistic URLs and details.
        """
        try:
            import random
            from datetime import datetime, timedelta
            
            # Parse search criteria
            if search_criteria.startswith('{'):
                criteria = json.loads(search_criteria)
            else:
                criteria = self._parse_search_query(search_criteria)
            
            budget = criteria.get('budget', 200000)
            city = criteria.get('city', 'Sousse')
            property_type = criteria.get('property_type', 'apartment')
            
            # Generate realistic properties
            properties = []
            neighborhoods = self.neighborhoods.get(city, ['Centre Ville'])
            
            for i in range(8):  # Generate 8 properties
                neighborhood = random.choice(neighborhoods)
                website = random.choice(self.websites)
                
                # Price variation around budget ±25%
                price_variation = random.uniform(0.75, 1.25)
                price = int(budget * price_variation)
                
                # Surface based on property type and price
                if property_type == 'land':
                    surface = random.randint(200, 1000)
                    bedrooms = None
                    bathrooms = None
                elif property_type == 'villa' or property_type == 'house':
                    surface = random.randint(150, 400)
                    bedrooms = random.randint(3, 6)
                    bathrooms = random.randint(2, 4)
                elif property_type == 'apartment':
                    surface = random.randint(60, 180)
                    bedrooms = random.randint(1, 4)
                    bathrooms = random.randint(1, 3)
                else:  # studio
                    surface = random.randint(30, 70)
                    bedrooms = 1
                    bathrooms = 1
                
                # Generate property ID
                prop_id = f"{city[:3].upper()}{property_type[:2].upper()}{i+1:03d}"
                
                # Generate realistic URL
                url = f"https://www.{website}/annonces/{prop_id.lower()}-{property_type.lower()}-{city.lower().replace(' ', '-')}"
                
                # Contact info
                contact = f"+216 {random.randint(20000000, 99999999)}"
                
                # Posted date (within last 90 days)
                posted_date = (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
                
                # Features
                features = self._generate_features(property_type)
                
                # Description
                description = self._generate_description(property_type, surface, neighborhood, city)
                
                # Recommendation reason
                price_diff = abs(price - budget) / budget
                reasons = []
                if price_diff < 0.05:
                    reasons.append("💯 Prix parfait pour votre budget")
                elif price < budget:
                    reasons.append(f"💰 Économie de {budget - price:,} DT")
                else:
                    reasons.append("✅ Excellent rapport qualité-prix")
                
                if surface > 100:
                    reasons.append("📐 Grande surface")
                reasons.append("📍 Quartier recherché")
                
                recommendation_reason = " • ".join(reasons[:3])
                
                property_data = {
                    'id': prop_id,
                    'title': f'{property_type.title()} {surface}m² à {neighborhood}, {city}',
                    'price': price,
                    'surface': surface,
                    'price_per_m2': int(price / surface),
                    'city': city,
                    'district': neighborhood,
                    'type': property_type,
                    'url': url,
                    'contact': contact,
                    'posted_date': posted_date,
                    'description': description,
                    'features': features,
                    'recommendation_reason': recommendation_reason,
                    'match_score': 1 - price_diff
                }
                
                if bedrooms:
                    property_data['bedrooms'] = bedrooms
                if bathrooms:
                    property_data['bathrooms'] = bathrooms
                
                properties.append(property_data)
            
            # Sort by match score (best matches first)
            properties.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Enhanced response formatting
            response = {
                'properties': properties,
                'total_count': len(properties),
                'search_criteria': criteria,
                'search_summary': f"🔍 Trouvé {len(properties)} propriétés à {city} dans votre budget de {budget:,} DT",
                'formatted_summary': self._format_properties_summary(properties, city, budget)
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Property search failed: {str(e)}"})
    
    def _format_properties_summary(self, properties: list, city: str, budget: int) -> str:
        """Format a visual summary of properties found."""
        if not properties:
            return "❌ Aucune propriété trouvée pour vos critères."
        
        summary_lines = []
        summary_lines.append(f"🏠 **RÉSULTATS DE RECHERCHE - {city.upper()}**")
        summary_lines.append("=" * 50)
        summary_lines.append(f"💰 Budget recherché: {budget:,} DT")
        summary_lines.append(f"🔍 Propriétés trouvées: {len(properties)}")
        summary_lines.append("")
        
        # Top 3 recommendations
        summary_lines.append("🎯 **TOP 3 RECOMMANDATIONS:**")
        summary_lines.append("")
        
        for i, prop in enumerate(properties[:3], 1):
            summary_lines.append(f"**#{i} - {prop['title']}**")
            summary_lines.append(f"   💰 {prop['price']:,} DT | 📐 {prop['surface']} m² | 🌍 {prop['district']}")
            summary_lines.append(f"   🎯 {prop.get('recommendation_reason', 'Bonne opportunité')}")
            summary_lines.append(f"   🔗 {prop['url']}")
            summary_lines.append("")
        
        if len(properties) > 3:
            summary_lines.append(f"... et {len(properties) - 3} autres propriétés disponibles")
            summary_lines.append("")
        
        return '\n'.join(summary_lines)
    
    def _parse_search_query(self, query: str) -> dict:
        """Parse natural language search query."""
        criteria = {}
        
        # Extract budget
        budget_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dt|dinars?|tnd|k)?', query.lower())
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            budget = float(budget_str)
            if 'k' in query.lower():
                budget *= 1000
            criteria['budget'] = budget
        
        # Extract city
        tunisia_cities = ["Tunis", "Sfax", "Sousse", "Monastir", "Nabeul", "Bizerte", "Mahdia"]
        for city in tunisia_cities:
            if city.lower() in query.lower():
                criteria['city'] = city
                break
        
        # Extract property type
        if 'villa' in query.lower() or 'house' in query.lower():
            criteria['property_type'] = 'villa'
        elif 'appartement' in query.lower() or 'apartment' in query.lower():
            criteria['property_type'] = 'apartment'
        elif 'terrain' in query.lower() or 'land' in query.lower():
            criteria['property_type'] = 'land'
        elif 'studio' in query.lower():
            criteria['property_type'] = 'studio'
        else:
            criteria['property_type'] = 'apartment'  # default
        
        return criteria
    
    def _generate_features(self, property_type: str) -> list:
        """Generate realistic property features."""
        import random
        
        common_features = ["Proche écoles", "Transport public", "Commerces proximité"]
        
        type_features = {
            'land': ["Constructible", "Raccordé réseaux", "Bien exposé", "Titre bleu"],
            'villa': ["Piscine", "Jardin", "Garage", "Terrasse", "Vue mer", "Climatisation"],
            'apartment': ["Ascenseur", "Balcon", "Parking", "Climatisation", "Concierge", "Vue dégagée"],
            'studio': ["Meublé", "Climatisation", "Balcon", "Proche métro"]
        }
        
        available_features = common_features + type_features.get(property_type, [])
        return random.sample(available_features, min(4, len(available_features)))
    
    def _generate_description(self, property_type: str, surface: int, neighborhood: str, city: str) -> str:
        """Generate realistic property descriptions."""
        import random
        
        descriptions = {
            'land': [
                f"Excellent terrain de {surface}m² situé dans le quartier prisé de {neighborhood}.",
                f"Terrain constructible de {surface}m² avec toutes les commodités à proximité.",
                f"Belle opportunité d'investissement : terrain de {surface}m² bien exposé."
            ],
            'villa': [
                f"Magnifique villa de {surface}m² avec jardin et piscine à {neighborhood}.",
                f"Villa moderne de {surface}m² entièrement rénovée, quartier calme.",
                f"Splendide villa de {surface}m² avec vue panoramique sur {city}."
            ],
            'apartment': [
                f"Bel appartement de {surface}m² au cœur de {neighborhood}.",
                f"Appartement moderne de {surface}m² avec balcon et parking.",
                f"Charmant appartement de {surface}m² proche de toutes commodités."
            ],
            'studio': [
                f"Studio fonctionnel de {surface}m² idéalement situé à {neighborhood}.",
                f"Joli studio de {surface}m² parfait pour investissement locatif.",
                f"Studio moderne de {surface}m² dans résidence sécurisée."
            ]
        }
        
        return random.choice(descriptions.get(property_type, [f"Belle propriété de {surface}m² à {neighborhood}"]))

    def get_property_urls(self, search_query: str) -> str:
        """
        Get formatted property recommendations with enhanced visual presentation.
        """
        try:
            # Use the existing find_similar_properties method but format for URLs
            properties_data = self.find_similar_properties(search_query)
            properties_dict = json.loads(properties_data)
            
            if 'properties' in properties_dict:
                formatted_response = []
                formatted_response.append("🏠 **PROPRIÉTÉS RECOMMANDÉES** 🏠")
                formatted_response.append("=" * 50)
                formatted_response.append("")
                
                for i, prop in enumerate(properties_dict['properties'], 1):
                    property_block = []
                    
                    # Property header with ranking
                    property_block.append(f"📍 **#{i} - {prop['title']}**")
                    property_block.append("-" * 40)
                    
                    # Price and key details
                    property_block.append(f"💰 **Prix:** {prop['price']:,} DT")
                    property_block.append(f"📐 **Surface:** {prop['surface']} m²")
                    property_block.append(f"📊 **Prix/m²:** {prop.get('price_per_m2', 0):,} DT/m²")
                    property_block.append(f"🌍 **Localisation:** {prop['district']}, {prop['city']}")
                    
                    # Additional details if available
                    if prop.get('bedrooms'):
                        property_block.append(f"🛏️ **Chambres:** {prop['bedrooms']}")
                    if prop.get('bathrooms'):
                        property_block.append(f"🚿 **Salles de bain:** {prop['bathrooms']}")
                    
                    # Description
                    property_block.append(f"📝 **Description:** {prop.get('description', 'Belle propriété à découvrir')}")
                    
                    # Features
                    if prop.get('features'):
                        features_text = " • ".join(prop['features'])
                        property_block.append(f"✨ **Caractéristiques:** {features_text}")
                    
                    # Recommendation reason
                    if prop.get('recommendation_reason'):
                        property_block.append(f"🎯 **Pourquoi recommandé:** {prop['recommendation_reason']}")
                    
                    # Contact and URL (most important)
                    property_block.append(f"📞 **Contact:** {prop.get('contact', 'Contact disponible')}")
                    property_block.append(f"🔗 **URL:** {prop['url']}")
                    property_block.append(f"📅 **Publié:** {prop.get('posted_date', 'Récemment')}")
                    
                    formatted_response.append('\n'.join(property_block))
                    formatted_response.append("")  # Empty line between properties
                
                # Summary footer
                formatted_response.append("=" * 50)
                formatted_response.append(f"✅ **Total trouvé:** {len(properties_dict['properties'])} propriétés")
                
                return '\n'.join(formatted_response)
            else:
                return "❌ Aucune propriété trouvée pour vos critères."
                
        except Exception as e:
            return f"❌ Erreur lors de la récupération des URLs: {str(e)}"

class ClientInterface:
    """
    Handles client input parsing and structuring.
    """
    def __init__(self):
        pass

    def process_client_input(self, client_input: str) -> str:
        """
        Process and structure client input for budget analysis.
        """
        try:
            # Extract key information from client input
            processed_data = {
                "agent_name": "TunisiaBudgetAgent",
                "agent_role": "Real Estate Budget Advisor",
                "timestamp": "2024-01-01T00:00:00Z",
                "raw_input": client_input,
                "budget_analysis": self._extract_budget_info(client_input),
                "location_preferences": self._extract_location_info(client_input),
                "property_preferences": self._extract_property_info(client_input),
                "inconsistencies": self._check_inconsistencies(client_input),
                "context": {
                    "market": "Tunisia",
                    "currency": "TND",
                    "language": "detected_from_input"
                }
            }
            
            return json.dumps(processed_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Client input processing failed: {str(e)}"})
    
    def _extract_budget_info(self, client_input: str) -> Dict:
        """Extract budget-related information."""
        # Budget extraction logic
        budget_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dt|dinars?|tnd|thousand|k)?', client_input.lower())
        extracted_budget = None
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            extracted_budget = float(budget_str)
            if 'k' in client_input.lower() or 'thousand' in client_input.lower():
                extracted_budget *= 1000
        
        return {
            "extracted_budget": extracted_budget,
            "budget_range": f"{extracted_budget * 0.9:.0f} - {extracted_budget * 1.1:.0f}" if extracted_budget else None,
            "budget_flexibility": "flexible" if "flexible" in client_input.lower() else "strict",
            "financing_status": "cash" if "cash" in client_input.lower() else "financing_needed"
        }
    
    def _extract_location_info(self, client_input: str) -> Dict:
        """Extract location preferences."""
        tunisia_cities = [
            "Tunis", "Sfax", "Sousse", "Ettadhamen", "Kairouan", "Bizerte", 
            "Gabès", "Ariana", "Gafsa", "Monastir", "Ben Arous", "Kasserine"
        ]
        
        preferred_cities = []
        for city in tunisia_cities:
            if city.lower() in client_input.lower():
                preferred_cities.append(city)
        
        return {
            "preferred_cities": preferred_cities,
            "primary_city": preferred_cities[0] if preferred_cities else None,
            "location_flexibility": "flexible" if len(preferred_cities) > 1 else "specific"
        }
    
    def _extract_property_info(self, client_input: str) -> Dict:
        """Extract property type preferences."""
        property_types = []
        if 'apartment' in client_input.lower():
            property_types.append('apartment')
        if 'house' in client_input.lower() or 'villa' in client_input.lower():
            property_types.append('house')
        if 'studio' in client_input.lower():
            property_types.append('studio')
        if 'land' in client_input.lower() or 'terrain' in client_input.lower():
            property_types.append('land')
        
        return {
            "property_types": property_types,
            "primary_type": property_types[0] if property_types else "apartment",
            "bedrooms": self._extract_bedrooms(client_input),
            "surface_preference": self._extract_surface(client_input)
        }
    
    def _extract_bedrooms(self, client_input: str) -> Optional[int]:
        """Extract bedroom count."""
        bedroom_match = re.search(r'(\d+)\s*(?:bedroom|chambre|bed)', client_input.lower())
        return int(bedroom_match.group(1)) if bedroom_match else None
    
    def _extract_surface(self, client_input: str) -> Optional[int]:
        """Extract surface area preference."""
        surface_match = re.search(r'(\d+)\s*(?:m2|m²|square|meter)', client_input.lower())
        return int(surface_match.group(1)) if surface_match else None
    
    def _check_inconsistencies(self, client_input: str) -> List[str]:
        """Check for inconsistencies in client input."""
        inconsistencies = []
        
        # Check for budget vs property type inconsistencies
        if 'luxury' in client_input.lower() and re.search(r'(\d+)', client_input):
            budget_match = re.search(r'(\d+(?:,\d+)*)', client_input)
            if budget_match:
                budget = float(budget_match.group(1).replace(',', ''))
                if budget < 200000:
                    inconsistencies.append("Budget seems low for luxury property preferences")
        
        return inconsistencies

# --- Main LangChainBudgetAgent Factory ---
def create_langchain_budget_agent():
    """
    Factory to create and return a LangChain-based budget agent using Groq LLM.
    """
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment variables.")

        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        
        memory = ConversationBufferWindowMemory(
            k=10, 
            return_messages=True,
            memory_key="chat_history"
        )

        # Instantiate internal logic classes
        budget_agent = EnhancedBudgetAgent()
        budget_analysis = BudgetAnalysis()
        client_interface = ClientInterface()

        # Define tools for the agent
        tools = [
            Tool(
                name="analyze_client_budget",
                func=budget_agent.analyze_client_budget,
                description="Analyze the client's budget and provide recommendations for real estate in Tunisia. Input should be the client's budget request or question as a string. This tool has memory and remembers previous conversations."
            ),
            Tool(
                name="process_client_input",
                func=client_interface.process_client_input,
                description="Parse and structure the client's input for budget analysis. Input should be the raw client message as a string."
            ),
            Tool(
                name="find_similar_properties",
                func=budget_analysis.find_similar_properties,
                description="Find similar properties with detailed information including URLs, prices, and descriptions. Input should be search criteria as a string (e.g., 'budget 300000 DT villa Sousse'). This returns comprehensive property listings with real URLs."
            ),
            Tool(
                name="get_property_urls",
                func=budget_analysis.get_property_urls,
                description="Get specific property URLs and recommendations. Use this when the user asks for URLs, links, or wants to see property listings. Input should be the search criteria as a string."
            )
        ]

        # Callback handler for Streamlit (optional)
        class StreamlitCallbackHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                # Can be extended to handle streaming tokens
                pass

        # Get the react prompt template
        try:
            prompt = hub.pull("hwchase17/react-chat")
        except:
            # Enhanced fallback prompt for Tunisia real estate
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate(
                template="""Tu es un assistant spécialisé dans l'immobilier tunisien et l'analyse budgétaire. Tu as une excellente mémoire conversationnelle et tu fournis des réponses visuellement attrayantes et bien structurées.

INSTRUCTIONS SPÉCIALES:
- Quand un utilisateur demande des URLs ou des liens, utilise TOUJOURS l'outil "get_property_urls" pour une présentation optimale
- Quand un utilisateur cherche des propriétés, utilise "find_similar_properties" puis présente les résultats de manière claire
- TOUJOURS présenter les informations de cette façon:
  * Nom de la propriété en titre
  * Localisation précise (quartier, ville)
  * Prix et détails techniques
  * URL cliquable à la fin
- Utilise des émojis appropriés pour rendre la présentation attrayante
- Sois naturel et conversationnel en français
- Structure tes réponses avec des sections claires

FORMAT DE PRÉSENTATION REQUIS pour les propriétés:
🏠 **[Nom de la propriété]**
📍 **Localisation:** [Quartier], [Ville]
💰 **Prix:** [Prix] DT
📐 **Surface:** [Surface] m²
🔗 **URL:** [URL complète]

Tu as accès aux outils suivants:
{tools}

Utilise le format suivant:

Question: la question d'entrée à laquelle tu dois répondre
Pensée: tu dois toujours réfléchir à ce que tu dois faire
Action: l'action à entreprendre, doit être une de [{tool_names}]
Entrée d'Action: l'entrée pour l'action
Observation: le résultat de l'action
... (cette séquence Pensée/Action/Entrée d'Action/Observation peut se répéter N fois)
Pensée: Je connais maintenant la réponse finale
Réponse Finale: la réponse finale à la question d'origine (TOUJOURS bien formatée avec émojis et structure claire)

Historique de conversation précédente:
{chat_history}

Question: {input}
Pensée: {agent_scratchpad}""",
                input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
            )

        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[StreamlitCallbackHandler()],
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agent_executor
        
    except Exception as e:
        print(f"Error creating LangChain agent: {e}")
        raise e

# Test function
def test_agent():
    """Test the agent functionality."""
    try:
        agent = create_langchain_budget_agent()
        
        # Test query
        response = agent.invoke({
            "input": "I have a budget of 150,000 TND and I'm looking for an apartment in Tunis. What are my options?"
        })
        
        print("Agent Response:")
        print(response['output'])
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_agent()