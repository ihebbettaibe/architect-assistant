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
            "property_type": None,
            "current_properties": [],
            "analysis_results": None
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create specialized tools for property analysis"""
        
        def search_properties_by_budget(budget: int, city: str = None, property_type: str = None) -> str:
            """Recherche des propriétés selon le budget et les préférences"""
            try:
                # Update context
                self.context["user_budget"] = budget
                if city:
                    self.context["preferred_city"] = city
                if property_type:
                    self.context["property_type"] = property_type
                
                # Create client profile for the existing analyze_client_budget method
                client_profile = {
                    "budget": budget,
                    "city": city or "Sousse",  # Default to Sousse if no city specified
                    "preferences": property_type or "habitation",
                    "min_size": 100,  # Default minimum size
                    "max_price": budget
                }
                
                # Use existing analyze_client_budget functionality
                analysis_result = self.base_agent.analyze_client_budget(client_profile)
                
                if analysis_result and analysis_result.get('comparable_properties'):
                    properties = analysis_result['comparable_properties']
                    
                    # Map properties to French keys for both display and context storage
                    mapped_properties = []
                    for prop in properties[:10]:  # Limit to top 10
                        mapped_prop = {
                            "prix": prop.get("Price", "N/A"),
                            "ville": prop.get("City", city),
                            "type": prop.get("Type", property_type),
                            "surface": prop.get("Surface", "N/A"),
                            "chambres": prop.get("Chambres", "N/A"),
                            "URL": prop.get("URL", ""),
                            "budget_fit_score": self._calculate_budget_fit_score(prop, budget)
                        }
                        mapped_properties.append(mapped_prop)
                    
                    # Store mapped properties in context
                    self.context["current_properties"] = mapped_properties
                    
                    # Format results for display
                    results = mapped_properties
                    
                    return f"Trouvé {len(properties)} propriétés dans votre budget. Voici les {min(10, len(properties))} meilleures options:\n{json.dumps(results, indent=2, ensure_ascii=False)}"
                else:
                    return f"Aucune propriété trouvée pour un budget de {budget} TND dans {city or 'toutes les villes'}."
            
            except Exception as e:
                return f"Erreur lors de la recherche: {str(e)}"
        
        def analyze_property_trends(city: str = None) -> str:
            """Analyse les tendances du marché immobilier"""
            try:
                analysis = self.budget_analysis.analyze_market_trends(city=city)
                self.context["analysis_results"] = analysis
                
                # Format key insights
                insights = []
                if "avg_price" in analysis:
                    insights.append(f"Prix moyen: {analysis['avg_price']:,.0f} TND")
                if "price_trend" in analysis:
                    insights.append(f"Tendance: {analysis['price_trend']}")
                if "popular_areas" in analysis:
                    insights.append(f"Zones populaires: {', '.join(analysis['popular_areas'][:3])}")
                
                return f"Analyse du marché pour {city or 'toutes les villes'}:\n" + "\n".join(insights)
            
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
            """Fournit des recommandations basées sur le budget"""
            try:
                recommendations = self.client_interface.get_budget_recommendations(budget)
                
                # Format recommendations
                formatted_recs = []
                for rec_type, details in recommendations.items():
                    if isinstance(details, dict):
                        formatted_recs.append(f"{rec_type}: {details}")
                    else:
                        formatted_recs.append(f"{rec_type}: {details}")
                
                return "Recommandations budgétaires:\n" + "\n".join(formatted_recs)
            
            except Exception as e:
                return f"Erreur lors des recommandations: {str(e)}"
        
        def get_property_details(property_index: int) -> str:
            """Obtient les détails d'une propriété spécifique"""
            try:
                if not self.context["current_properties"]:
                    return "Aucune propriété disponible. Effectuez d'abord une recherche."
                
                if 0 <= property_index < len(self.context["current_properties"]):
                    prop = self.context["current_properties"][property_index]
                    
                    details = []
                    for key, value in prop.items():
                        if key not in ["embedding", "metadata"]:
                            details.append(f"{key}: {value}")
                    
                    return f"Détails de la propriété {property_index + 1}:\n" + "\n".join(details)
                else:
                    return f"Index invalide. Propriétés disponibles: 0-{len(self.context['current_properties'])-1}"
            
            except Exception as e:
                return f"Erreur lors de la récupération des détails: {str(e)}"
        
        # Create tool objects
        tools = [
            StructuredTool.from_function(
                func=search_properties_by_budget,
                name="search_properties",
                description="Recherche des propriétés selon le budget, la ville et le type de propriété"
            ),
            StructuredTool.from_function(
                func=analyze_property_trends,
                name="analyze_trends",
                description="Analyse les tendances du marché immobilier pour une ville donnée"
            ),
            StructuredTool.from_function(
                func=compare_properties,
                name="compare_properties",
                description="Compare les propriétés trouvées selon différents critères (price, surface, rooms)"
            ),
            StructuredTool.from_function(
                func=get_budget_recommendations,
                name="budget_recommendations",
                description="Fournit des recommandations personnalisées basées sur le budget"
            ),
            StructuredTool.from_function(
                func=get_property_details,
                name="property_details",
                description="Obtient les détails complets d'une propriété spécifique par son index"
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools and memory"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un agent immobilier expert spécialisé dans l'analyse budgétaire et la recherche de propriétés en Tunisie.

Tu as accès à une base de données de propriétés immobilières et à des outils d'analyse spécialisés.

INSTRUCTIONS IMPORTANTES:
1. Toujours commencer par comprendre le budget et les préférences de l'utilisateur
2. Utiliser les outils appropriés pour rechercher et analyser les propriétés
3. Fournir des réponses détaillées et personnalisées en français
4. Garder en mémoire le contexte de la conversation
5. Être proactif dans les recommandations
6. Utiliser un langage professionnel mais accessible
7. Ne jamais montrer les appels d'outils ou erreurs techniques à l'utilisateur
8. TOUJOURS interpréter et présenter les résultats des outils de manière claire et utile

OUTILS DISPONIBLES:
- search_properties: Pour rechercher des propriétés selon le budget et critères
- analyze_trends: Pour analyser les tendances du marché immobilier
- compare_properties: Pour comparer différentes propriétés
- budget_recommendations: Pour fournir des conseils budgétaires personnalisés
- property_details: Pour obtenir des détails spécifiques d'une propriété

COMPORTEMENT:
- Réponds toujours en français
- Sois utile, informatif et professionnel
- Cache les détails techniques et les erreurs système
- Focus sur les besoins de l'utilisateur
- Propose des solutions concrètes
- Après avoir utilisé un outil, présente TOUJOURS les résultats de manière claire et structurée
- Ne montre JAMAIS les appels de fonction bruts"""),
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
            # Add callback handler if provided
            callbacks = [callback_handler] if callback_handler else []
            
            # Prepare the complete input with context embedded in the message
            context_info = []
            if self.context.get("user_budget"):
                context_info.append(f"Budget: {self.context['user_budget']:,} TND")
            if self.context.get("preferred_city"):
                context_info.append(f"Ville: {self.context['preferred_city']}")
            if self.context.get("property_type"):
                context_info.append(f"Type: {self.context['property_type']}")
            
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
            "property_type": None,
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
        
        if self.context.get("property_type"):
            summary_parts.append(f"Type: {self.context['property_type']}")
        
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
