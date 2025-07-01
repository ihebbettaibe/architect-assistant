"""
LangChain-based Budget Agent using Groq API
Integrates with property database and provides conversational real estate budget analysis
"""

import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import existing budget functionality
from budget_agent_base import EnhancedBudgetAgent
from budget_analysis import BudgetAnalysis
from client_interface import ClientInterface

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit integration"""
    
    def __init__(self, container=None):
        self.container = container
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        if self.container:
            self.container.info(f"🔧 Utilisation de l'outil: {serialized.get('name', 'Unknown')}")
    
    def on_tool_end(self, output: str, **kwargs):
        if self.container:
            self.container.success("✅ Outil exécuté avec succès")

class LangChainBudgetAgent:
    """
    Advanced budget agent using LangChain with Groq API
    Combines conversational AI with specialized property analysis tools
    """
    
    def __init__(self, groq_api_key: str, model_name: str = "mixtral-8x7b-32768", data_folder: str = None, use_couchdb: bool = True):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Determine the correct data folder path
        if data_folder is None:
            # Try different possible paths for the data folder
            script_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(script_dir, "../../cleaned_data"),  # From agents/budget/ to root/cleaned_data
                os.path.join(script_dir, "../../../cleaned_data"),  # Alternative path
                "cleaned_data",  # If running from root
                os.path.abspath(os.path.join(script_dir, "../../cleaned_data"))  # Absolute path
            ]
            
            data_folder = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_folder = path
                    break
            
            if data_folder is None:
                raise FileNotFoundError(
                    "Data folder 'cleaned_data' not found. Please ensure the cleaned_data folder exists with property CSV files."
                )
        
        # Initialize base agent functionality with CouchDB or data folder
        # Create a combined agent class like in the working native app
        class FullBudgetAgent(EnhancedBudgetAgent, BudgetAnalysis, ClientInterface):
            pass
        
        if use_couchdb:
            try:
                self.base_agent = FullBudgetAgent(use_couchdb=True)
                print("✅ LangChain agent initialized with CouchDB")
            except Exception as e:
                print(f"⚠️ CouchDB initialization failed, falling back to CSV: {e}")
                if data_folder is None:
                    # Try different possible paths for the data folder
                    script_dir = os.path.dirname(__file__)
                    possible_paths = [
                        os.path.join(script_dir, "../../cleaned_data"),  # From agents/budget/ to root/cleaned_data
                        os.path.join(script_dir, "../../../cleaned_data"),  # Alternative path
                        "cleaned_data",  # If running from root
                        os.path.abspath(os.path.join(script_dir, "../../cleaned_data"))  # Absolute path
                    ]
                    
                    data_folder = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            data_folder = path
                            break
                    
                    if data_folder is None:
                        raise FileNotFoundError(
                            "Data folder 'cleaned_data' not found. Please ensure the cleaned_data folder exists with property CSV files."
                        )
                
                self.base_agent = FullBudgetAgent(data_folder=data_folder, use_couchdb=False)
        else:
            # Use CSV files
            if data_folder is None:
                # Try different possible paths for the data folder
                script_dir = os.path.dirname(__file__)
                possible_paths = [
                    os.path.join(script_dir, "../../cleaned_data"),  # From agents/budget/ to root/cleaned_data
                    os.path.join(script_dir, "../../../cleaned_data"),  # Alternative path
                    "cleaned_data",  # If running from root
                    os.path.abspath(os.path.join(script_dir, "../../cleaned_data"))  # Absolute path
                ]
                
                data_folder = None
                for path in possible_paths:
                    if os.path.exists(path):
                        data_folder = path
                        break
                
                if data_folder is None:
                    raise FileNotFoundError(
                        "Data folder 'cleaned_data' not found. Please ensure the cleaned_data folder exists with property CSV files."
                    )
            
            self.base_agent = FullBudgetAgent(data_folder=data_folder, use_couchdb=False)
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
        # Context storage
        self.context = {
            "user_budget": None,
            "preferred_city": None,
            "current_properties": [],
            "analysis_results": None
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create specialized tools for property analysis"""
        
        def search_properties_by_budget(budget: int, city: str = None) -> str:
            """Recherche hybride de terrains utilisant l'analyse complète de l'agent standard"""
            try:
                print(f"🔍 Hybrid tool called: search_properties_by_budget(budget={budget}, city={city})")
                
                # Update context
                self.context["user_budget"] = budget
                if city:
                    self.context["preferred_city"] = city
                
                # Use standard agent's comprehensive analysis - always search for terrain
                client_profile = {
                    "city": city or "Sousse",
                    "budget": budget,
                    "preferences": "terrain habitation construction",
                    "min_size": 100,
                    "max_price": budget
                }
                
                # Get full analysis from standard agent (including AI recommendations)
                budget_analysis = self.base_agent.analyze_client_budget(client_profile)
                
                if budget_analysis and budget_analysis.get('comparable_properties'):
                    properties = budget_analysis['comparable_properties']
                    market_stats = budget_analysis['market_statistics']
                    ai_analysis = budget_analysis['budget_analysis']
                    
                    # Store enhanced properties with all data
                    enhanced_properties = []
                    for prop in properties[:10]:
                        enhanced_prop = {
                            # Preserve all original data with both French and English keys
                            **prop,
                            "prix": prop.get("Price", 0),
                            "ville": prop.get("Location", prop.get("City", city)),
                            "type": prop.get("Type", property_type),
                            "surface": prop.get("Surface", 0),
                            "chambres": prop.get("Chambres", "N/A"),
                            "titre": prop.get("Title", f"Propriété {property_type}"),
                            "localisation": prop.get("Location", prop.get("City", city)),
                            "url": prop.get("URL", ""),
                            "Price": prop.get("Price", 0),
                            "Location": prop.get("Location", prop.get("City", city)),
                            "Type": prop.get("Type", property_type),
                            "Surface": prop.get("Surface", 0),
                            "Chambres": prop.get("Chambres", "N/A"),
                            "Title": prop.get("Title", f"Propriété {property_type}"),
                            "URL": prop.get("URL", ""),
                            "price_per_m2": prop.get("price_per_m2", prop.get("Price", 0) / max(prop.get("Surface", 1), 1)),
                            "budget_fit_score": self._calculate_budget_fit_score(prop, budget)
                        }
                        enhanced_properties.append(enhanced_prop)
                    
                    # Store comprehensive analysis results
                    self.context["current_properties"] = enhanced_properties
                    self.context["analysis_results"] = {
                        "market_statistics": market_stats,
                        "ai_analysis": ai_analysis,
                        "budget_analysis": budget_analysis
                    }
                    
                    print(f"✅ Found {len(properties)} properties with comprehensive analysis for budget {budget} in {city}")
                    
                    # Create comprehensive response combining LangChain and standard agent insights
                    response_parts = [
                        f"🏠 **Analyse Complète Terminée**",
                        f"J'ai trouvé **{len(properties)} propriétés** correspondant à votre budget de **{budget:,} DT** à {city}.",
                        "",
                        f"📊 **Statistiques du Marché:**",
                        f"• Prix moyen: {market_stats['price_stats']['mean']:,.0f} DT",
                        f"• Fourchette: {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT",
                        f"• Faisabilité budgétaire: {market_stats['budget_feasibility']['feasibility_ratio']:.1%}",
                        "",
                        f"🎯 **Recommandations AI:**",
                        f"• {ai_analysis.get('budget_validation', 'Budget analysé')}",
                        f"• {ai_analysis.get('recommendations', 'Recommandations générées')}",
                        f"• Niveau de confiance: {ai_analysis.get('confidence_score', 0.8):.1%}",
                        ""
                    ]
                    
                    # Add market position if available
                    if ai_analysis.get('market_position'):
                        response_parts.append(f"📈 **Position sur le Marché:** {ai_analysis['market_position']}")
                        response_parts.append("")
                    
                    # Add negotiation tips if available
                    if ai_analysis.get('price_negotiation_tips'):
                        response_parts.append(f"💡 **Conseils de Négociation:** {ai_analysis['price_negotiation_tips']}")
                        response_parts.append("")
                    
                    # Add investment analysis if available  
                    if ai_analysis.get('investment_potential'):
                        response_parts.append(f"📈 **Potentiel d'Investissement:** {ai_analysis['investment_potential']}")
                        response_parts.append("")
                    
                    # Add market trends if available
                    if market_stats.get('market_trends'):
                        response_parts.append(f"📊 **Tendances du Marché:** {market_stats['market_trends']}")
                        response_parts.append("")
                    
                    response_parts.append(f"🏆 **Top 5 Propriétés Recommandées:**")
                    
                    # Add top 5 properties with details
                    for i, prop in enumerate(enhanced_properties[:5], 1):
                        response_parts.extend([
                            f"",
                            f"**{i}. {prop.get('Title', 'Propriété')}**",
                            f"   💰 Prix: {prop.get('Price', 0):,.0f} DT",
                            f"   📐 Surface: {prop.get('Surface', 0):.0f} m²",
                            f"   💵 Prix/m²: {prop.get('price_per_m2', 0):,.0f} DT/m²",
                            f"   📍 Localisation: {prop.get('Location', 'N/A')}",
                            f"   🎯 Score: {prop.get('budget_fit_score', 0):.1f}/10",
                            f"   🔗 URL: {prop.get('URL', 'Non disponible')}"
                        ])
                    
                    return "\n".join(response_parts)
                else:
                    print(f"❌ No properties found for budget {budget} in {city}")
                    
                    # Use standard agent's AI recommendations even when no properties found
                    empty_profile_analysis = self.base_agent.process_client_input(
                        f"Je cherche une propriété avec un budget de {budget} DT à {city}"
                    )
                    
                    response_parts = [
                        f"❌ **Aucune propriété trouvée** pour votre budget de **{budget:,} DT** à {city}.",
                        "",
                        f"💡 **Recommandations de l'IA:**"
                    ]
                    
                    if empty_profile_analysis.get('suggestions'):
                        for suggestion in empty_profile_analysis['suggestions']:
                            response_parts.append(f"• {suggestion}")
                    
                    if empty_profile_analysis.get('targeted_questions'):
                        response_parts.extend([
                            "",
                            f"❓ **Questions pour Affiner:**"
                        ])
                        for question in empty_profile_analysis['targeted_questions']:
                            response_parts.append(f"• {question}")
                    
                    return "\n".join(response_parts)
            
            except Exception as e:
                return f"❌ Erreur lors de l'analyse hybride: {str(e)}"
        
        def analyze_property_trends(city: str = None) -> str:
            """Analyse les tendances du marché immobilier avec des insights AI avancés"""
            try:
                # Use standard agent for comprehensive market analysis
                if hasattr(self.base_agent, 'process_client_input'):
                    market_query = f"Analyse les tendances du marché immobilier à {city or 'toutes les villes de Tunisie'}"
                    analysis_result = self.base_agent.process_client_input(market_query)
                    
                    if analysis_result:
                        # Extract insights from standard agent analysis
                        insights = []
                        
                        if analysis_result.get('suggestions'):
                            insights.append("📊 **Tendances Identifiées:**")
                            for suggestion in analysis_result['suggestions'][:3]:
                                insights.append(f"• {suggestion}")
                        
                        if analysis_result.get('targeted_questions'):
                            insights.append("\n❓ **Points d'Attention:**")
                            for question in analysis_result['targeted_questions'][:2]:
                                insights.append(f"• {question}")
                        
                        if analysis_result.get('reliability_score'):
                            confidence = analysis_result['reliability_score']
                            insights.append(f"\n🎯 **Fiabilité de l'analyse:** {confidence:.1%}")
                        
                        return "\n".join(insights) if insights else f"Analyse du marché pour {city or 'toutes les villes'} en cours..."
                
                # Fallback to basic analysis
                if hasattr(self, 'property_data') and self.property_data is not None:
                    city_data = self.property_data
                    if city:
                        city_data = city_data[city_data['City'].str.contains(city, case=False, na=False)]
                    
                    if not city_data.empty:
                        avg_price = city_data['Price'].mean()
                        price_trend = "stable"  # Simplified trend analysis
                        popular_areas = city_data['Location'].value_counts().head(3).index.tolist()
                        
                        insights = [
                            f"📊 **Analyse du marché pour {city or 'toutes les villes'}:**",
                            f"• Prix moyen: {avg_price:,.0f} DT",
                            f"• Tendance: {price_trend}",
                            f"• Zones populaires: {', '.join(popular_areas)}"
                        ]
                        
                        return "\n".join(insights)
                
                return f"Analyse du marché pour {city or 'toutes les villes'} - données insuffisantes"
            
            except Exception as e:
                return f"Erreur lors de l'analyse: {str(e)}"
        
        def compare_properties(criteria: str = "price") -> str:
            """Compare les propriétés actuelles selon un critère"""
            try:
                if not self.context["current_properties"]:
                    return "Aucune propriété à comparer. Effectuez d'abord une recherche."
                
                properties = self.context["current_properties"]
                
                if criteria.lower() in ["surface", "size"]:
                    sorted_props = sorted(properties, key=lambda x: float(x.get("surface", 0)), reverse=True)
                    comparison_key = "surface"
                    unit = "m²"
                elif criteria.lower() in ["price", "prix"]:
                    sorted_props = sorted(properties, key=lambda x: float(x.get("prix", 0)))
                    comparison_key = "prix"
                    unit = "TND"
                elif criteria.lower() in ["rooms", "chambres"]:
                    sorted_props = sorted(properties, key=lambda x: int(x.get("chambres", 0)), reverse=True)
                    comparison_key = "chambres"
                    unit = ""
                else:
                    sorted_props = sorted(properties, key=lambda x: x.get("budget_fit_score", 0), reverse=True)
                    comparison_key = "budget_fit_score"
                    unit = "/10"
                
                results = []
                for i, prop in enumerate(sorted_props[:5], 1):
                    value = prop.get(comparison_key, "N/A")
                    results.append(f"{i}. {prop.get('ville', 'N/A')} - {prop.get('type', 'N/A')}: {value}{unit}")
                
                return f"Comparaison par {criteria}:\n" + "\n".join(results)
            
            except Exception as e:
                return f"Erreur lors de la comparaison: {str(e)}"
        
        def get_budget_recommendations(budget: int) -> str:
            """Fournit des recommandations basées sur le budget avec analyse IA avancée"""
            try:
                # Use standard agent's comprehensive budget analysis
                if hasattr(self.base_agent, 'process_client_input'):
                    budget_query = f"J'ai un budget de {budget} DT pour un projet immobilier. Quelles sont mes options et recommandations?"
                    budget_analysis = self.base_agent.process_client_input(budget_query)
                    
                    recommendations = []
                    
                    if budget_analysis:
                        # Extract budget from the analysis
                        extracted_budget = budget_analysis.get('budget_analysis', {}).get('extracted_budget')
                        if extracted_budget:
                            recommendations.append(f"💰 **Budget Analysé:** {extracted_budget:,} DT")
                        
                        # Add confidence level
                        confidence = budget_analysis.get('confidence_level', 'medium')
                        reliability = budget_analysis.get('reliability_score', 0.5)
                        recommendations.append(f"🎯 **Niveau de Confiance:** {confidence.title()} ({reliability:.1%})")
                        
                        # Add targeted suggestions
                        if budget_analysis.get('suggestions'):
                            recommendations.append("\n💡 **Recommandations Personnalisées:**")
                            for i, suggestion in enumerate(budget_analysis['suggestions'], 1):
                                recommendations.append(f"{i}. {suggestion}")
                        
                        # Add market feasibility insights
                        if budget_analysis.get('next_actions'):
                            recommendations.append("\n🎯 **Actions Recommandées:**")
                            action_descriptions = {
                                'search_comparable_properties': 'Rechercher des propriétés comparables dans votre gamme de prix',
                                'validate_budget_with_market_data': 'Valider votre budget avec les données actuelles du marché',
                                'collect_more_budget_info': 'Préciser vos contraintes budgétaires et vos préférences',
                                'clarify_financing_options': 'Explorer vos options de financement bancaire'
                            }
                            
                            for action in budget_analysis['next_actions']:
                                description = action_descriptions.get(action, action)
                                recommendations.append(f"• {description}")
                        
                        # Add inconsistencies if any
                        if budget_analysis.get('inconsistencies_detected'):
                            recommendations.append("\n⚠️ **Points d'Attention:**")
                            for inconsistency in budget_analysis['inconsistencies_detected']:
                                recommendations.append(f"• {inconsistency.get('message', 'Point à vérifier')}")
                        
                        # Add targeted questions for further refinement
                        if budget_analysis.get('targeted_questions'):
                            recommendations.append("\n❓ **Questions pour Affiner votre Projet:**")
                            for question in budget_analysis['targeted_questions'][:3]:
                                recommendations.append(f"• {question}")
                    
                    return "\n".join(recommendations) if recommendations else f"Analyse du budget de {budget:,} DT en cours..."
                
                # Fallback to basic recommendations
                recommendations = [
                    f"💰 **Recommandations pour votre budget de {budget:,} DT:**",
                    "",
                    f"📊 **Gamme Accessible:**"
                ]
                
                if budget < 100000:
                    recommendations.extend([
                        "• Terrain constructible en périphérie",
                        "• Appartement studio ou 2 pièces",
                        "• Projet de rénovation",
                        "💡 Conseil: Considérez un financement bancaire pour élargir vos options"
                    ])
                elif budget < 300000:
                    recommendations.extend([
                        "• Villa économique 3-4 pièces",
                        "• Appartement familial bien situé", 
                        "• Terrain + construction modulaire",
                        "💡 Conseil: Excellent budget pour un premier achat immobilier"
                    ])
                elif budget < 500000:
                    recommendations.extend([
                        "• Villa moderne avec jardin",
                        "• Duplex ou triplex",
                        "• Propriété d'investissement locatif",
                        "💡 Conseil: Budget confortable permettant de bons choix"
                    ])
                else:
                    recommendations.extend([
                        "• Villa de prestige",
                        "• Propriété avec piscine",
                        "• Investissement immobilier multiple",
                        "💡 Conseil: Budget élevé, focus sur la localisation premium"
                    ])
                
                return "\n".join(recommendations)
            
            except Exception as e:
                return f"Erreur lors des recommandations: {str(e)}"
        
        def get_property_details(property_index: int) -> str:
            """Obtient les détails d'une propriété spécifique avec analyse complète"""
            try:
                if not self.context["current_properties"]:
                    return "Aucune propriété disponible. Effectuez d'abord une recherche."
                
                if 0 <= property_index < len(self.context["current_properties"]):
                    prop = self.context["current_properties"][property_index]
                    
                    # Get comprehensive property analysis
                    details = [
                        f"🏠 **Analyse Détaillée - Propriété {property_index + 1}**",
                        f"",
                        f"📋 **Informations de Base:**",
                        f"• Titre: {prop.get('Title', 'N/A')}",
                        f"• Prix: {prop.get('Price', 0):,.0f} DT",
                        f"• Surface: {prop.get('Surface', 0):.0f} m²",
                        f"• Prix/m²: {prop.get('price_per_m2', 0):,.0f} DT/m²",
                        f"• Localisation: {prop.get('Location', 'N/A')}",
                        f"• Type: {prop.get('Type', 'N/A')}",
                        f"",
                        f"📊 **Analyse Budgétaire:**",
                        f"• Score de compatibilité: {prop.get('budget_fit_score', 0):.1f}/10",
                    ]
                    
                    # Add budget comparison if available
                    if self.context.get("user_budget"):
                        user_budget = self.context["user_budget"]
                        prop_price = prop.get('Price', 0)
                        if prop_price > 0:
                            budget_ratio = prop_price / user_budget
                            if budget_ratio <= 1.0:
                                details.append(f"• Dans votre budget: ✅ ({budget_ratio:.1%} du budget)")
                            else:
                                details.append(f"• Dépasse votre budget: ❌ (+{(budget_ratio-1)*100:.1f}%)")
                    
                    # Add additional details if available
                    if prop.get('Chambres') and prop.get('Chambres') != 'N/A':
                        details.extend([
                            f"",
                            f"🏡 **Caractéristiques:**",
                            f"• Chambres: {prop.get('Chambres')}",
                        ])
                        
                        # Add other details if available
                        if prop.get('salles_de_bain'):
                            details.append(f"• Salles de bain: {prop.get('salles_de_bain')}")
                        if prop.get('parking'):
                            details.append(f"• Parking: {prop.get('parking')}")
                        if prop.get('jardin'):
                            details.append(f"• Jardin: {prop.get('jardin')}")
                    
                    # Add market analysis from stored context
                    if self.context.get("analysis_results"):
                        market_stats = self.context["analysis_results"].get("market_statistics", {})
                        if market_stats:
                            mean_price = market_stats.get('price_stats', {}).get('mean', 0)
                            if mean_price > 0:
                                price_vs_market = (prop.get('Price', 0) / mean_price - 1) * 100
                                details.extend([
                                    f"",
                                    f"📈 **Position sur le Marché:**",
                                    f"• Prix vs moyenne du marché: {price_vs_market:+.1f}%",
                                ])
                                
                                if price_vs_market < -10:
                                    details.append("• 💚 Très bon rapport qualité-prix")
                                elif price_vs_market < 0:
                                    details.append("• 💛 Rapport qualité-prix correct")
                                elif price_vs_market < 10:
                                    details.append("• 🟡 Prix dans la moyenne")
                                else:
                                    details.append("• 🔴 Prix élevé par rapport au marché")
                    
                    # Add URL if available
                    if prop.get('URL') and prop.get('URL') != 'Non disponible':
                        details.extend([
                            f"",
                            f"🔗 **Lien vers l'annonce:**",
                            f"{prop.get('URL')}"
                        ])
                    
                    return "\n".join(details)
                else:
                    return f"Index invalide. Propriétés disponibles: 0-{len(self.context['current_properties'])-1}"
            
            except Exception as e:
                return f"Erreur lors de la récupération des détails: {str(e)}"
        
        def analyze_user_question(question: str) -> str:
            """Analyse intelligente des questions utilisateur avec l'IA de l'agent standard"""
            try:
                # Use standard agent's intelligent question processing
                if hasattr(self.base_agent, 'process_client_input'):
                    # Add context to the question for better analysis
                    context_info = []
                    if self.context.get("user_budget"):
                        context_info.append(f"Budget: {self.context['user_budget']:,} DT")
                    if self.context.get("preferred_city"):
                        context_info.append(f"Ville: {self.context['preferred_city']}")
                    
                    context_str = " | ".join(context_info) if context_info else ""
                    enhanced_question = f"{question}\nContexte: {context_str}" if context_str else question
                    
                    # Process with standard agent
                    analysis_result = self.base_agent.process_client_input(enhanced_question)
                    
                    if analysis_result:
                        response_parts = []
                        
                        # Add extracted budget if found 
                        budget_info = analysis_result.get('budget_analysis', {})
                        if budget_info.get('extracted_budget'):
                            budget = budget_info['extracted_budget']
                            response_parts.append(f"💰 **Budget identifié:** {budget:,} DT")
                            # Update context
                            self.context["user_budget"] = budget
                        
                        # Add confidence assessment
                        confidence = analysis_result.get('confidence_level', 'medium')
                        reliability = analysis_result.get('reliability_score', 0.5)
                        response_parts.append(f"🎯 **Analyse:** Confiance {confidence} ({reliability:.1%})")
                        
                        # Add suggestions
                        if analysis_result.get('suggestions'):
                            response_parts.append("\n💡 **Réponses et Suggestions:**")
                            for suggestion in analysis_result['suggestions']:
                                response_parts.append(f"• {suggestion}")
                        
                        # Add targeted questions for clarification
                        if analysis_result.get('targeted_questions'):
                            response_parts.append("\n❓ **Questions pour Préciser:**")
                            for targeted_q in analysis_result['targeted_questions'][:2]:
                                response_parts.append(f"• {targeted_q}")
                        
                        # Add next actions
                        if analysis_result.get('next_actions'):
                            response_parts.append("\n🎯 **Actions Recommandées:**")
                            action_descriptions = {
                                'search_comparable_properties': 'Rechercher des propriétés dans votre gamme',
                                'validate_budget_with_market_data': 'Valider avec les données du marché',
                                'collect_more_budget_info': 'Préciser vos contraintes financières',
                                'clarify_financing_options': 'Explorer les options de financement'
                            }
                            
                            for action in analysis_result['next_actions']:
                                description = action_descriptions.get(action, action)
                                response_parts.append(f"• {description}")
                        
                        return "\n".join(response_parts) if response_parts else "Analyse en cours..."
                
                # Fallback response
                return f"Analyse de votre question: '{question}' - Besoin de plus de contexte pour une réponse précise."
            
            except Exception as e:
                return f"Erreur lors de l'analyse: {str(e)}"
        
        # Create tool objects with enhanced descriptions
        tools = [
            StructuredTool.from_function(
                func=search_properties_by_budget,
                name="search_properties",
                description="Recherche complète de propriétés avec analyse de marché, recommandations IA et statistiques détaillées selon le budget, la ville et le type"
            ),
            StructuredTool.from_function(
                func=analyze_property_trends,
                name="analyze_trends",
                description="Analyse avancée des tendances du marché immobilier avec insights IA et recommandations stratégiques"
            ),
            StructuredTool.from_function(
                func=compare_properties,
                name="compare_properties",
                description="Comparaison intelligente des propriétés trouvées selon différents critères (prix, surface, chambres, score)"
            ),
            StructuredTool.from_function(
                func=get_budget_recommendations,
                name="budget_recommendations",
                description="Recommandations budgétaires personnalisées avec analyse IA complète et plan d'action"
            ),
            StructuredTool.from_function(
                func=get_property_details,
                name="property_details",
                description="Analyse détaillée d'une propriété spécifique avec évaluation de marché et compatibilité budgétaire"
            ),
            StructuredTool.from_function(
                func=analyze_user_question,
                name="analyze_question",
                description="Analyse intelligente des questions utilisateur avec extraction d'informations et recommandations contextuelles"
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools and memory"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
    ("system", """
Tu es un agent immobilier expert en Tunisie, spécialisé dans l'analyse budgétaire et la recherche de propriétés.

Tu combines capacités conversationnelles et outils d'analyse IA.

Règle clé : utilise TOUJOURS les outils disponibles pour des données réelles. Ne jamais inventer d'infos.

Workflow :
1. Comprends le contexte (budget, ville, préférences)
2. Utilise systématiquement search_properties pour les recherches
3. Analyse les tendances avec analyze_trends
4. Donne des conseils via budget_recommendations
5. Traite les questions complexes avec analyze_question
6. Présente des résultats clairs, structurés et actionnables

Outils :
- search_properties : recherche et scoring IA
- analyze_trends : insights marché
- compare_properties : comparaison multi-critères
- budget_recommendations : conseils stratégiques
- property_details : analyse détaillée
- analyze_question : questions complexes

Comportement :
- Réponds en français professionnel accessible
- Base-toi toujours sur les outils pour des données factuelles
- Combine analyse quantitative et qualitative
- Adapte selon le client, cache les détails techniques
- Propose solutions concrètes avec justifications

Présente résultats avec sections claires et métriques, inclut URLs, résume points clés et actions.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

        
        # Create agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def chat(self, message: str, callback_handler=None) -> Dict[str, Any]:
        """
        Process a user message and return response with context
        """
        try:
            print(f"💬 LangChain agent processing: {message}")
            
            # Add callback handler if provided
            callbacks = [callback_handler] if callback_handler else []
            
            # Prepare the complete input with context embedded in the message
            context_info = []
            if self.context.get("user_budget"):
                context_info.append(f"Budget: {self.context['user_budget']:,} TND")
            if self.context.get("preferred_city"):
                context_info.append(f"Ville: {self.context['preferred_city']}")
          
            context_str = " | ".join(context_info) if context_info else "Aucun contexte défini"
            
            # Enhanced message with context
            enhanced_message = f"{message}\n\nContexte actuel: {context_str}"
            
            # Run the agent with only the input key (as expected by LangChain)
            response = self.agent_executor.invoke(
                {"input": enhanced_message},
                config={"callbacks": callbacks}
            )
            
            return {
                "response": response.get("output", "Désolé, je n'ai pas pu traiter votre demande."),
                "context": self.context.copy(),
                "properties": self.context.get("current_properties", []),
                "memory": self.get_conversation_history()
            }
        
        except Exception as e:
            # Return a user-friendly error message instead of technical details
            error_msg = "Je rencontre des difficultés techniques. Veuillez réessayer ou reformuler votre question."
            
            return {
                "response": error_msg,
                "context": self.context.copy(),
                "properties": [],
                "memory": self.get_conversation_history(),
                "error": str(e)  # Keep technical error for debugging, but don't show to user
            }
    
    def clear_memory(self):
        """Clear conversation memory and context"""
        if hasattr(self.memory, 'clear'):
            self.memory.clear()
        elif hasattr(self.memory, 'chat_memory'):
            self.memory.chat_memory.clear()
        
        # Reset context but keep user preferences
        user_budget = self.context.get("user_budget")
        preferred_city = self.context.get("preferred_city")
        
        self.context = {
            "user_budget": user_budget,
            "preferred_city": preferred_city,
            "current_properties": [],
            "analysis_results": None
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []
        
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.context = {
            "user_budget": None,
            "preferred_city": None,
            "current_properties": [],
            "analysis_results": None
        }
    
    def update_context(self, **kwargs):
        """Update agent context"""
        self.context.update(kwargs)
    
    def get_context_summary(self) -> str:
        """Get a formatted summary of current context"""
        summary_parts = []
        
        if self.context.get("user_budget"):
            summary_parts.append(f"Budget: {self.context['user_budget']:,} TND")
        
        if self.context.get("preferred_city"):
            summary_parts.append(f"Ville: {self.context['preferred_city']}")
        
        if self.context.get("current_properties"):
            summary_parts.append(f"Propriétés trouvées: {len(self.context['current_properties'])}")
        
        return " | ".join(summary_parts) if summary_parts else "Aucun contexte défini"
    
    def _calculate_budget_fit_score(self, property_data: dict, user_budget: int) -> float:
        """Calculate how well a property fits the user's budget (0-10 scale)"""
        try:
            property_price = property_data.get("Price", 0)
            
            if property_price == 0 or user_budget == 0:
                return 5.0  # Neutral score if no price data
            
            # Calculate price difference ratio
            price_ratio = property_price / user_budget
            
            # Score calculation:
            # - Perfect match (price == budget): 10
            # - Under budget: score decreases gradually
            # - Over budget: score decreases more rapidly
            if price_ratio <= 1.0:
                # Property is within or under budget
                score = 10.0 - (1.0 - price_ratio) * 2.0
            else:
                # Property is over budget
                over_budget_penalty = (price_ratio - 1.0) * 5.0
                score = max(1.0, 10.0 - over_budget_penalty)
            
            return round(min(10.0, max(1.0, score)), 1)
            
        except Exception as e:
            return 5.0  # Default neutral score on error

# Factory function for creating agent instances
def create_langchain_budget_agent(groq_api_key: str, model_name: str = "mixtral-8x7b-32768", data_folder: str = None, use_couchdb: bool = True) -> LangChainBudgetAgent:
    """Factory function to create a LangChain budget agent"""
    return LangChainBudgetAgent(groq_api_key=groq_api_key, model_name=model_name, data_folder=data_folder, use_couchdb=use_couchdb)
