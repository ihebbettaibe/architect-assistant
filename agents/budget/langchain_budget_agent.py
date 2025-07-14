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
    Handles budget analysis and property search logic.
    """
    def __init__(self):
        self.tunisia_cities = [
            "Tunis", "Sfax", "Sousse", "Ettadhamen", "Kairouan", "Bizerte", 
            "Gabès", "Ariana", "Gafsa", "Monastir", "Ben Arous", "Kasserine",
            "Médenine", "Nabeul", "Tataouine", "Beja", "Jendouba", "Mahdia",
            "Sidi Bouzid", "Siliana", "Manouba", "Kef", "Tozeur", "Zaghouan", "Kebili"
        ]
        
    def analyze_client_budget(self, client_input: str) -> str:
        """
        Analyze client budget from string input and return JSON string.
        """
        try:
            # Parse the client input to extract budget information
            client_profile = self._parse_client_input(client_input)
            
            # Perform budget analysis
            budget_amount = client_profile.get('budget', 0)
            city = client_profile.get('city', 'Unknown')
            property_type = client_profile.get('property_type', 'apartment')
            
            # Simulate market analysis based on Tunisia real estate market
            market_stats = self._get_market_statistics(city, property_type, budget_amount)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(client_profile, market_stats)
            
            analysis_result = {
                'client_info': client_profile,
                'market_statistics': market_stats,
                'recommendations': recommendations,
                'budget_feasibility': self._assess_budget_feasibility(budget_amount, market_stats)
            }
            
            return json.dumps(analysis_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Budget analysis failed: {str(e)}"})
    
    def _parse_client_input(self, client_input: str) -> Dict:
        """Parse client input to extract structured information."""
        # Extract budget using regex
        budget_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dt|dinars?|tnd|thousand|k)?', client_input.lower())
        budget = 0
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            budget = float(budget_str)
            # Handle thousands notation
            if 'k' in client_input.lower() or 'thousand' in client_input.lower():
                budget *= 1000
        
        # Extract city
        city = None
        for tunisia_city in self.tunisia_cities:
            if tunisia_city.lower() in client_input.lower():
                city = tunisia_city
                break
        
        # Extract property type
        property_type = 'apartment'  # default
        if 'house' in client_input.lower() or 'villa' in client_input.lower():
            property_type = 'house'
        elif 'studio' in client_input.lower():
            property_type = 'studio'
        elif 'land' in client_input.lower() or 'terrain' in client_input.lower():
            property_type = 'land'
        
        return {
            'budget': budget,
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
    Handles market and property data analysis.
    """
    def __init__(self):
        pass

    def find_similar_properties(self, search_criteria: str) -> str:
        """
        Find similar properties based on search criteria.
        """
        try:
            # Parse search criteria
            criteria = json.loads(search_criteria) if search_criteria.startswith('{') else {'query': search_criteria}
            
            # Simulate property search results
            properties = [
                {
                    'id': 1,
                    'title': 'Modern Apartment in Tunis',
                    'price': 180000,
                    'surface': 85,
                    'price_per_m2': 2118,
                    'city': 'Tunis',
                    'district': 'El Menzah',
                    'type': 'apartment',
                    'bedrooms': 2,
                    'bathrooms': 1
                },
                {
                    'id': 2,
                    'title': 'Villa in Sousse',
                    'price': 350000,
                    'surface': 200,
                    'price_per_m2': 1750,
                    'city': 'Sousse',
                    'district': 'Kantaoui',
                    'type': 'house',
                    'bedrooms': 4,
                    'bathrooms': 3
                },
                {
                    'id': 3,
                    'title': 'Studio in Sfax',
                    'price': 95000,
                    'surface': 45,
                    'price_per_m2': 2111,
                    'city': 'Sfax',
                    'district': 'Centre Ville',
                    'type': 'studio',
                    'bedrooms': 1,
                    'bathrooms': 1
                }
            ]
            
            return json.dumps({
                'properties': properties,
                'total_count': len(properties),
                'search_criteria': criteria
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Property search failed: {str(e)}"})

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
                description="Analyze the client's budget and provide recommendations for real estate in Tunisia. Input should be the client's budget request or question as a string."
            ),
            Tool(
                name="process_client_input",
                func=client_interface.process_client_input,
                description="Parse and structure the client's input for budget analysis. Input should be the raw client message as a string."
            ),
            Tool(
                name="find_similar_properties",
                func=budget_analysis.find_similar_properties,
                description="Find similar properties using market data. Input should be search criteria as a string or JSON."
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
            # Fallback prompt if hub is not available
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate(
                template="""You are a helpful assistant specialized in Tunisia real estate budget analysis.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}""",
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