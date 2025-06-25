# LangGraph Implementation Draft for Budget Agent

## Overview
This document outlines how to integrate LangGraph into the existing budget agent project to create a more sophisticated, stateful workflow for real estate budget analysis.

## 1. Dependencies to Install

```bash
pip install langgraph langchain langchain-openai langchain-community
```

Add to `requirements.txt`:
```
langgraph>=0.1.0
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.1.0
```

## 2. Project Structure Changes

```
agents/
├── budget/
│   ├── langgraph_workflow/
│   │   ├── __init__.py
│   │   ├── budget_graph.py          # Main graph definition
│   │   ├── nodes.py                 # Individual workflow nodes
│   │   ├── state.py                 # State management
│   │   └── tools.py                 # LangGraph tools
│   ├── budget_agent_base.py
│   ├── budget_analysis.py
│   └── budget_streamlit_app.py
```

## 3. State Definition

Create `agents/budget/langgraph_workflow/state.py`:

```python
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import MessagesState

class BudgetAnalysisState(MessagesState):
    """State for the budget analysis workflow."""
    
    # User input and context
    user_input: str
    conversation_history: List[Dict[str, Any]]
    conversation_context: Dict[str, Any]
    
    # Budget analysis
    extracted_budget: Optional[float]
    budget_confidence: str  # 'high', 'medium', 'low'
    budget_components: Dict[str, Any]
    
    # Market analysis
    market_data: Optional[Dict[str, Any]]
    comparable_properties: List[Dict[str, Any]]
    market_statistics: Optional[Dict[str, Any]]
    
    # User preferences
    preferred_city: Optional[str]
    property_type: Optional[str]
    min_size: Optional[float]
    max_price: Optional[float]
    
    # Analysis results
    recommendations: List[str]
    next_actions: List[str]
    clarifying_questions: List[str]
    inconsistencies: List[Dict[str, Any]]
    
    # Workflow control
    current_step: str
    needs_clarification: bool
    analysis_complete: bool
    error_message: Optional[str]
```

## 4. Workflow Nodes

Create `agents/budget/langgraph_workflow/nodes.py`:

```python
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from .state import BudgetAnalysisState
import re
import json

class BudgetWorkflowNodes:
    """Individual nodes for the budget analysis workflow."""
    
    def __init__(self, budget_agent):
        self.budget_agent = budget_agent
    
    def input_analysis_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Analyze user input and extract intent."""
        user_input = state["user_input"]
        
        # Extract budget information
        budget_info = self._extract_budget_info(user_input)
        
        # Update state
        state["extracted_budget"] = budget_info.get("amount")
        state["budget_confidence"] = budget_info.get("confidence", "low")
        state["budget_components"] = budget_info.get("components", {})
        
        # Extract location preferences
        city = self._extract_city(user_input)
        if city:
            state["preferred_city"] = city
        
        # Extract property preferences
        property_info = self._extract_property_preferences(user_input)
        state.update(property_info)
        
        # Determine next step
        if state["extracted_budget"] and state["budget_confidence"] in ["high", "medium"]:
            state["current_step"] = "market_analysis"
            state["needs_clarification"] = False
        else:
            state["current_step"] = "clarification"
            state["needs_clarification"] = True
        
        return state
    
    def clarification_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Generate clarifying questions when information is unclear."""
        questions = []
        
        if not state.get("extracted_budget"):
            questions.append("Quel est votre budget total pour ce projet immobilier ?")
        
        if not state.get("preferred_city"):
            questions.append("Dans quelle ville ou région souhaitez-vous investir ?")
        
        if not state.get("property_type"):
            questions.append("Quel type de bien recherchez-vous (terrain, appartement, villa, etc.) ?")
        
        if state["budget_confidence"] == "low" and state.get("extracted_budget"):
            questions.append("Votre budget inclut-il les frais de notaire et d'agence ?")
        
        state["clarifying_questions"] = questions
        state["current_step"] = "waiting_clarification"
        
        return state
    
    def market_analysis_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Perform market analysis based on extracted information."""
        try:
            # Prepare client profile
            client_profile = {
                "budget": state.get("extracted_budget"),
                "city": state.get("preferred_city", "Sousse"),
                "property_type": state.get("property_type", "terrain"),
                "min_size": state.get("min_size", 150),
                "max_price": state.get("max_price") or state.get("extracted_budget")
            }
            
            # Perform market analysis using existing agent
            market_analysis = self.budget_agent.analyze_client_budget(client_profile)
            
            if market_analysis:
                state["market_data"] = market_analysis
                state["comparable_properties"] = market_analysis.get("comparable_properties", [])
                state["market_statistics"] = market_analysis.get("market_statistics", {})
                state["current_step"] = "recommendation_generation"
            else:
                state["current_step"] = "error_handling"
                state["error_message"] = "Impossible d'obtenir les données du marché"
        
        except Exception as e:
            state["current_step"] = "error_handling"
            state["error_message"] = str(e)
        
        return state
    
    def recommendation_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Generate recommendations based on analysis."""
        recommendations = []
        next_actions = []
        
        budget = state.get("extracted_budget", 0)
        market_stats = state.get("market_statistics", {})
        
        if market_stats:
            avg_price = market_stats.get("price_stats", {}).get("mean", 0)
            min_price = market_stats.get("price_stats", {}).get("min", 0)
            
            # Budget feasibility analysis
            if budget >= avg_price:
                recommendations.append(f"Votre budget de {budget:,.0f} DT est bien adapté au marché actuel")
                next_actions.append("search_premium_properties")
            elif budget >= min_price:
                recommendations.append("Votre budget permet d'accéder à certaines propriétés du marché")
                next_actions.append("search_comparable_properties")
            else:
                recommendations.append("Votre budget pourrait nécessiter un ajustement")
                next_actions.append("suggest_budget_optimization")
            
            # Property recommendations
            if len(state.get("comparable_properties", [])) > 0:
                recommendations.append(f"{len(state['comparable_properties'])} propriétés correspondent à vos critères")
                next_actions.append("show_property_details")
        
        state["recommendations"] = recommendations
        state["next_actions"] = next_actions
        state["current_step"] = "finalization"
        
        return state
    
    def finalization_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Finalize the analysis and prepare response."""
        state["analysis_complete"] = True
        state["current_step"] = "complete"
        
        # Add summary message
        summary_parts = []
        if state.get("extracted_budget"):
            summary_parts.append(f"Budget analysé: {state['extracted_budget']:,.0f} DT")
        if state.get("preferred_city"):
            summary_parts.append(f"Ville: {state['preferred_city']}")
        if state.get("comparable_properties"):
            summary_parts.append(f"{len(state['comparable_properties'])} propriétés trouvées")
        
        summary = " • ".join(summary_parts)
        state["messages"].append(AIMessage(content=f"Analyse terminée: {summary}"))
        
        return state
    
    def error_handling_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
        """Handle errors in the workflow."""
        error_msg = state.get("error_message", "Erreur inconnue")
        
        state["recommendations"] = [f"Une erreur s'est produite: {error_msg}"]
        state["next_actions"] = ["retry_analysis"]
        state["current_step"] = "error"
        
        return state
    
    def _extract_budget_info(self, text: str) -> Dict[str, Any]:
        """Extract budget information from text."""
        # Regex patterns for budget detection
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:k|mille|thousand)',  # 150k, 150 mille
            r'(\d+(?:\.\d+)?)\s*(?:m|million)',          # 1.5m, 1.5 million
            r'(\d{1,3}(?:[,\s]\d{3})*)\s*(?:dt|dinar|dinars|TND)',  # 150,000 DT
            r'budget.*?(\d{1,3}(?:[,\s]\d{3})*)',        # budget de 150,000
            r'(\d{1,3}(?:[,\s]\d{3})*)\s*(?:pour|de|€|$)',  # 150,000 pour
        ]
        
        budget_info = {"amount": None, "confidence": "low", "components": {}}
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '').replace(' ', '')
                try:
                    amount = float(amount_str)
                    
                    # Handle units
                    if 'k' in match.group(0).lower() or 'mille' in match.group(0).lower():
                        amount *= 1000
                    elif 'm' in match.group(0).lower() or 'million' in match.group(0).lower():
                        amount *= 1000000
                    
                    budget_info["amount"] = amount
                    budget_info["confidence"] = "high" if "budget" in text.lower() else "medium"
                    break
                except ValueError:
                    continue
        
        return budget_info
    
    def _extract_city(self, text: str) -> Optional[str]:
        """Extract city from text."""
        tunisia_cities = [
            'tunis', 'sfax', 'sousse', 'kairouan', 'bizerte', 
            'mahdia', 'monastir', 'nabeul', 'ariana', 'ben arous'
        ]
        
        text_lower = text.lower()
        for city in tunisia_cities:
            if city in text_lower:
                return city.title()
        return None
    
    def _extract_property_preferences(self, text: str) -> Dict[str, Any]:
        """Extract property preferences from text."""
        preferences = {}
        
        # Property types
        if any(word in text.lower() for word in ['terrain', 'land', 'lot']):
            preferences["property_type"] = "terrain"
        elif any(word in text.lower() for word in ['appartement', 'apartment', 'flat']):
            preferences["property_type"] = "appartement"
        elif any(word in text.lower() for word in ['villa', 'maison', 'house']):
            preferences["property_type"] = "villa"
        
        # Size preferences
        size_patterns = [
            r'(\d+)\s*(?:m²|m2|mètres?\s*carrés?)',
            r'(\d+)\s*(?:metres?\s*carres?)',
        ]
        
        for pattern in size_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    size = float(match.group(1))
                    preferences["min_size"] = size
                    break
                except ValueError:
                    continue
        
        return preferences
```

## 5. Main Graph Definition

Create `agents/budget/langgraph_workflow/budget_graph.py`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import BudgetAnalysisState
from .nodes import BudgetWorkflowNodes

class BudgetAnalysisGraph:
    """Main LangGraph workflow for budget analysis."""
    
    def __init__(self, budget_agent):
        self.budget_agent = budget_agent
        self.nodes = BudgetWorkflowNodes(budget_agent)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph."""
        
        # Create the graph
        workflow = StateGraph(BudgetAnalysisState)
        
        # Add nodes
        workflow.add_node("input_analysis", self.nodes.input_analysis_node)
        workflow.add_node("clarification", self.nodes.clarification_node)
        workflow.add_node("market_analysis", self.nodes.market_analysis_node)
        workflow.add_node("recommendation", self.nodes.recommendation_node)
        workflow.add_node("finalization", self.nodes.finalization_node)
        workflow.add_node("error_handling", self.nodes.error_handling_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "input_analysis")
        
        # Conditional routing from input_analysis
        workflow.add_conditional_edges(
            "input_analysis",
            self._route_after_input_analysis,
            {
                "clarification": "clarification",
                "market_analysis": "market_analysis",
                "error": "error_handling"
            }
        )
        
        # From clarification, either go to market_analysis or end
        workflow.add_conditional_edges(
            "clarification",
            self._route_after_clarification,
            {
                "wait": END,
                "continue": "market_analysis"
            }
        )
        
        # From market_analysis
        workflow.add_conditional_edges(
            "market_analysis",
            self._route_after_market_analysis,
            {
                "recommendation": "recommendation",
                "error": "error_handling"
            }
        )
        
        # Linear flow for final steps
        workflow.add_edge("recommendation", "finalization")
        workflow.add_edge("finalization", END)
        workflow.add_edge("error_handling", END)
        
        # Compile the graph
        return workflow.compile(checkpointer=self.memory)
    
    def _route_after_input_analysis(self, state: BudgetAnalysisState) -> str:
        """Route after input analysis based on extracted information."""
        if state.get("error_message"):
            return "error"
        elif state.get("needs_clarification", True):
            return "clarification"
        else:
            return "market_analysis"
    
    def _route_after_clarification(self, state: BudgetAnalysisState) -> str:
        """Route after clarification questions."""
        if state.get("clarifying_questions"):
            return "wait"  # Need to wait for user response
        else:
            return "continue"
    
    def _route_after_market_analysis(self, state: BudgetAnalysisState) -> str:
        """Route after market analysis."""
        if state.get("error_message"):
            return "error"
        else:
            return "recommendation"
    
    async def process_input(self, user_input: str, conversation_context: dict = None) -> dict:
        """Process user input through the workflow."""
        
        # Prepare initial state
        initial_state = {
            "user_input": user_input,
            "conversation_context": conversation_context or {},
            "conversation_history": conversation_context.get("history", []) if conversation_context else [],
            "current_step": "input_analysis",
            "needs_clarification": False,
            "analysis_complete": False,
            "messages": []
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": "budget_analysis_session"}}
        result = await self.graph.ainvoke(initial_state, config=config)
        
        # Format response
        response = {
            "budget_analysis": {
                "extracted_budget": result.get("extracted_budget"),
                "confidence": result.get("budget_confidence", "low"),
                "components": result.get("budget_components", {})
            },
            "market_data": result.get("market_data"),
            "comparable_properties": result.get("comparable_properties", []),
            "recommendations": result.get("recommendations", []),
            "next_actions": result.get("next_actions", []),
            "clarifying_questions": result.get("clarifying_questions", []),
            "inconsistencies": result.get("inconsistencies", []),
            "analysis_complete": result.get("analysis_complete", False),
            "confidence_level": result.get("budget_confidence", "low"),
            "reliability_score": self._calculate_reliability_score(result)
        }
        
        return response
    
    def _calculate_reliability_score(self, state: BudgetAnalysisState) -> float:
        """Calculate reliability score based on analysis quality."""
        score = 0.0
        
        # Budget extraction quality
        if state.get("extracted_budget"):
            if state.get("budget_confidence") == "high":
                score += 0.4
            elif state.get("budget_confidence") == "medium":
                score += 0.2
        
        # Market data availability
        if state.get("market_data"):
            score += 0.3
        
        # Property matches
        if state.get("comparable_properties"):
            score += 0.2
        
        # Completeness
        if state.get("analysis_complete"):
            score += 0.1
        
        return min(score, 1.0)
```

## 6. Integration with Streamlit App

Modify the existing `budget_streamlit_app.py`:

```python
# Add these imports at the top
from agents.budget.langgraph_workflow.budget_graph import BudgetAnalysisGraph

# Modify the FullBudgetAgent class
class FullBudgetAgent(EnhancedBudgetAgent, BudgetAnalysis, ClientInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langgraph_workflow = BudgetAnalysisGraph(self)
    
    async def process_client_input_with_langgraph(self, user_input: str, conversation_context: dict = None):
        """Process input using LangGraph workflow."""
        return await self.langgraph_workflow.process_input(user_input, conversation_context)

# Modify the execute_analysis_with_progress function
async def execute_analysis_with_progress_langgraph(user_input):
    """Execute analysis using LangGraph workflow."""
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Initialisation du workflow...")
        progress_bar.progress(10)
        
        status_text.text("🧠 Analyse de votre demande...")
        progress_bar.progress(30)
        
        # Use LangGraph workflow
        result = await st.session_state.agent.process_client_input_with_langgraph(
            user_input, 
            st.session_state.conversation_context
        )
        
        status_text.text("💰 Analyse du budget...")
        progress_bar.progress(50)
        
        status_text.text("📊 Traitement des données de marché...")
        progress_bar.progress(70)
        
        status_text.text("💡 Génération des recommandations...")
        progress_bar.progress(90)
        
        status_text.text("✅ Finalisation de l'analyse...")
        progress_bar.progress(100)
        
        progress_bar.empty()
        status_text.empty()        
    return result
```

## 7. Advanced Features to Add Later

### A. Human-in-the-Loop for Clarifications
```python
def add_human_feedback_node(self, state: BudgetAnalysisState) -> BudgetAnalysisState:
    """Handle human feedback and continue workflow."""
    # This would integrate with Streamlit's session state
    # to pause workflow and wait for user input
    pass
```

### B. Memory and Conversation Persistence
```python
from langgraph.checkpoint.postgres import PostgresSaver

# Use PostgreSQL for persistent memory across sessions
memory = PostgresSaver.from_conn_string("postgresql://...")
```

### C. Tool Integration
```python
from langchain_core.tools import tool

@tool
def search_properties_tool(city: str, budget: float, property_type: str) -> list:
    """Search for properties matching criteria."""
    # Integration with your existing property search logic
    pass

@tool
def calculate_financing_tool(price: float, down_payment: float) -> dict:
    """Calculate financing options."""
    # Financial calculations
    pass
```

### D. Multi-Agent Collaboration
```python
# Add specialized sub-agents for different tasks
class MarketAnalysisAgent:
    """Specialized agent for market analysis."""
    pass

class FinancingAgent:
    """Specialized agent for financing calculations."""
    pass
```

## 8. Benefits of LangGraph Implementation

1. **Stateful Conversations**: Maintains context across interactions
2. **Conditional Logic**: Smart routing based on analysis results
3. **Error Handling**: Graceful handling of errors and edge cases
4. **Scalability**: Easy to add new nodes and workflows
5. **Debugging**: Visual graph representation for debugging
6. **Memory**: Persistent conversation memory
7. **Human-in-the-Loop**: Natural integration points for user feedback

## 9. Migration Steps

1. **Phase 1**: Install dependencies and create basic graph structure
2. **Phase 2**: Migrate core analysis logic to LangGraph nodes
3. **Phase 3**: Update Streamlit integration
4. **Phase 4**: Add advanced features (memory, tools, etc.)
5. **Phase 5**: Testing and optimization

## 10. Testing Strategy

```python
# Create test cases for the workflow
async def test_budget_extraction():
    graph = BudgetAnalysisGraph(mock_agent)
    result = await graph.process_input("Mon budget est de 150,000 DT pour Sousse")
    assert result["budget_analysis"]["extracted_budget"] == 150000
    assert result["budget_analysis"]["confidence"] == "high"

async def test_clarification_flow():
    graph = BudgetAnalysisGraph(mock_agent)
    result = await graph.process_input("Je cherche une propriété")
    assert len(result["clarifying_questions"]) > 0
```

This implementation will provide a much more sophisticated and maintainable workflow system for your budget agent!
