import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv

# Import specialized agents
from budget import FullBudgetAgent
# from style_agent import StyleAgent
# from Project_agent import ProjectAgent
# from Regulation_agent import RegulationAgent
# from context_agent import ContextAgent

load_dotenv()

class ArchitectureAssistantOrchestrator:
    """
    Agent orchestrateur - Main coordinator for the architecture assistant system
    
    Manages conversation flow, delegates tasks to specialized agents,
    and compiles results to generate a coherent architectural brief.
    """
    
    def __init__(self):
        """Initialize the orchestrator and all specialized agents"""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.conversation_history = []
        self.conversation_context = {
            "client_profile": {},
            "project_details": {},
            "budget_info": {},
            "style_preferences": {},
            "regulatory_context": {},
            "current_phase": "initial_contact"
        }
          # Initialize specialized agents
        self.agents = {
            "budget": FullBudgetAgent(data_folder="cleaned_data"),
            "style": None,  # StyleAgent(),
            "project": None,  # ProjectAgent(),
            "regulation": None,  # RegulationAgent(),
            "context": None  # ContextAgent()
        }
        
        # Conversation phases and their priorities
        self.conversation_phases = {
            "initial_contact": ["context", "project"],
            "project_definition": ["project", "budget"],
            "budget_validation": ["budget", "regulation"],
            "style_exploration": ["style", "context"],
            "regulatory_check": ["regulation", "budget"],
            "brief_compilation": ["all"]
        }
    
    def process_client_message(self, message: str) -> Dict[str, Any]:
        """
        Main entry point - processes client message and coordinates agent responses
        
        Args:
            message: Natural language message from client
            
        Returns:
            Structured response with agent outputs and next steps
        """
        # Add message to conversation history
        self.conversation_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "role": "client",
            "content": message
        })
        
        # Analyze message intent and determine which agents to invoke
        message_analysis = self._analyze_message_intent(message)
        
        # Route to appropriate agents
        agent_responses = self._route_to_agents(message, message_analysis)
        
        # Update conversation context
        self._update_conversation_context(agent_responses)
        
        # Generate coordinated response
        orchestrated_response = self._generate_orchestrated_response(agent_responses, message_analysis)
        
        # Add response to conversation history
        self.conversation_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "role": "assistant",
            "content": orchestrated_response
        })
        
        # Determine next phase and actions
        next_phase = self._determine_next_phase()
        
        return {
            "orchestrator_response": orchestrated_response,
            "agent_responses": agent_responses,
            "conversation_context": self.conversation_context,
            "current_phase": self.conversation_context["current_phase"],
            "next_phase": next_phase,
            "conversation_completeness": self._assess_conversation_completeness(),
            "suggested_questions": self._get_suggested_questions(),
            "architectural_brief": self._generate_partial_brief() if self._is_ready_for_brief() else None
        }
    
    def _analyze_message_intent(self, message: str) -> Dict[str, Any]:
        """Analyze client message to determine intent and relevant domains"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Analyze the client message and determine which architectural domains are relevant.
                        Return JSON with:
                        {
                            "primary_intent": "budget/style/project_type/regulation/context/general",
                            "relevant_agents": ["budget", "style", "project", "regulation", "context"],
                            "urgency": "high/medium/low",
                            "information_type": "new_info/clarification/question",
                            "confidence": 0.0-1.0
                        }
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this client message: {message}"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Message analysis error: {e}")
            return {
                "primary_intent": "general",
                "relevant_agents": ["context"],
                "urgency": "medium",
                "information_type": "new_info",
                "confidence": 0.5
            }
    
    def _route_to_agents(self, message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate specialized agents"""
        agent_responses = {}
        
        # Always include context agent for client profiling
        relevant_agents = analysis.get("relevant_agents", [])
        if "context" not in relevant_agents:
            relevant_agents.append("context")
        
        # Route to each relevant agent
        for agent_name in relevant_agents:
            if agent_name in self.agents and self.agents[agent_name] is not None:
                try:
                    if agent_name == "budget":
                        response = self.agents[agent_name].process_client_input(
                            message, self.conversation_context
                        )
                        agent_responses[agent_name] = response
                    # Add other agents when implemented
                    # elif agent_name == "style":
                    #     response = self.agents[agent_name].process_client_input(
                    #         message, self.conversation_context
                    #     )
                    #     agent_responses[agent_name] = response
                except Exception as e:
                    print(f"Error routing to {agent_name} agent: {e}")
                    agent_responses[agent_name] = {"error": str(e)}
        
        return agent_responses
    
    def _update_conversation_context(self, agent_responses: Dict[str, Any]):
        """Update conversation context based on agent responses"""
        for agent_name, response in agent_responses.items():
            if agent_name == "budget" and "budget_analysis" in response:
                self.conversation_context["budget_info"].update(response["budget_analysis"])
            elif agent_name == "style" and "style_analysis" in response:
                self.conversation_context["style_preferences"].update(response["style_analysis"])
            elif agent_name == "project" and "project_analysis" in response:
                self.conversation_context["project_details"].update(response["project_analysis"])
            elif agent_name == "regulation" and "regulatory_analysis" in response:
                self.conversation_context["regulatory_context"].update(response["regulatory_analysis"])
            elif agent_name == "context" and "client_profile" in response:
                self.conversation_context["client_profile"].update(response["client_profile"])
    
    def _generate_orchestrated_response(self, agent_responses: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate a coordinated response from multiple agent outputs"""
        try:
            # Compile agent insights
            insights = []
            questions = []
            suggestions = []
            
            for agent_name, response in agent_responses.items():
                if "targeted_questions" in response:
                    questions.extend(response["targeted_questions"])
                if "suggestions" in response:
                    suggestions.extend(response["suggestions"])
                if "inconsistencies_detected" in response:
                    for inconsistency in response["inconsistencies_detected"]:
                        insights.append(f"⚠️ {inconsistency['message']}")
            
            # Generate natural language response
            context_for_response = {
                "agent_insights": insights,
                "suggested_questions": questions[:3],  # Limit to 3 most relevant
                "recommendations": suggestions[:2],  # Limit to 2 main suggestions
                "current_phase": self.conversation_context["current_phase"]
            }
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are a friendly architectural assistant coordinating multiple specialists.
                        Generate a natural, conversational response that:
                        1. Acknowledges the client's input
                        2. Shares relevant insights from specialists
                        3. Asks 1-2 most important follow-up questions
                        4. Provides helpful suggestions
                        Keep the tone professional but warm, and avoid overwhelming the client.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Generate response based on: {json.dumps(context_for_response)}"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "Je vous remercie pour ces informations. Pouvez-vous me donner plus de détails sur votre projet ?"
    
    def _determine_next_phase(self) -> str:
        """Determine the next conversation phase based on current context"""
        current_phase = self.conversation_context["current_phase"]
        
        # Phase progression logic
        if current_phase == "initial_contact":
            if self.conversation_context["project_details"]:
                return "project_definition"
        elif current_phase == "project_definition":
            if self.conversation_context["budget_info"]:
                return "budget_validation"
        elif current_phase == "budget_validation":
            if self.conversation_context["budget_info"].get("reliability_score", 0) > 0.7:
                return "style_exploration"
        elif current_phase == "style_exploration":
            if self.conversation_context["style_preferences"]:
                return "regulatory_check"
        elif current_phase == "regulatory_check":
            return "brief_compilation"
        
        return current_phase
    
    def _assess_conversation_completeness(self) -> Dict[str, float]:
        """Assess how complete each domain of information is"""
        completeness = {}
        
        # Budget completeness
        budget_info = self.conversation_context["budget_info"]
        budget_score = 0.0
        if budget_info.get("extracted_budget"):
            budget_score += 0.4
        if budget_info.get("financing_status") != "unknown":
            budget_score += 0.3
        if budget_info.get("budget_flexibility") != "unknown":
            budget_score += 0.3
        completeness["budget"] = budget_score
        
        # Project completeness
        project_info = self.conversation_context["project_details"]
        project_score = 0.0
        if project_info.get("project_type"):
            project_score += 0.5
        if project_info.get("surface"):
            project_score += 0.3
        if project_info.get("timeline"):
            project_score += 0.2
        completeness["project"] = project_score
        
        # Overall completeness
        completeness["overall"] = sum(completeness.values()) / len(completeness)
        
        return completeness
    
    def _get_suggested_questions(self) -> List[str]:
        """Get suggested questions for the current phase"""
        current_phase = self.conversation_context["current_phase"]
        
        questions = {
            "initial_contact": [
                "Quel type de projet architectural souhaitez-vous réaliser ?",
                "Avez-vous déjà une idée de votre budget ?",
                "Dans quelle région envisagez-vous ce projet ?"
            ],
            "project_definition": [
                "Quelle superficie envisagez-vous pour votre projet ?",
                "Quel est votre délai souhaité pour la réalisation ?",
                "Avez-vous déjà un terrain ou un bien existant ?"
            ],
            "budget_validation": [
                "Votre budget inclut-il l'achat du terrain ?",
                "Avez-vous une solution de financement ?",
                "Quelle marge avez-vous pour les imprévus ?"
            ],
            "style_exploration": [
                "Quel style architectural vous inspire ?",
                "Avez-vous des références ou images qui vous plaisent ?",
                "Quelles sont vos priorités en termes d'espaces ?"
            ]
        }
        
        return questions.get(current_phase, [])
    
    def _is_ready_for_brief(self) -> bool:
        """Check if enough information is available to generate a brief"""
        completeness = self._assess_conversation_completeness()
        return completeness["overall"] >= 0.7
    
    def _generate_partial_brief(self) -> Dict[str, Any]:
        """Generate architectural brief from collected information"""
        return {
            "brief_id": f"brief_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "client_profile": self.conversation_context["client_profile"],
            "project_summary": {
                "type": self.conversation_context["project_details"].get("project_type"),
                "surface": self.conversation_context["project_details"].get("surface"),
                "budget": self.conversation_context["budget_info"].get("extracted_budget"),
                "timeline": self.conversation_context["project_details"].get("timeline")
            },
            "budget_analysis": self.conversation_context["budget_info"],
            "style_preferences": self.conversation_context["style_preferences"],
            "regulatory_considerations": self.conversation_context["regulatory_context"],
            "recommendations": self._compile_recommendations(),
            "completeness_score": self._assess_conversation_completeness()["overall"],
            "generated_at": pd.Timestamp.now().isoformat()
        }
    
    def _compile_recommendations(self) -> List[str]:
        """Compile recommendations from all agents"""
        recommendations = []
        
        # Add budget recommendations
        budget_info = self.conversation_context["budget_info"]
        if budget_info.get("extracted_budget"):
            recommendations.append(f"Budget estimé: {budget_info['extracted_budget']:,.0f} DT")
        
        # Add other recommendations as agents are implemented
        
        return recommendations

# Example usage
if __name__ == "__main__":
    print("🏗️ Initializing Architecture Assistant Orchestrator...")
    orchestrator = ArchitectureAssistantOrchestrator()
    
    # Simulate conversation
    test_messages = [
        "Bonjour, je souhaite construire une maison d'environ 150m² avec un budget de 350000 DT",
        "C'est pour une construction neuve, et j'ai déjà le terrain",
        "Le budget est assez flexible, je peux aller jusqu'à 400000 DT si nécessaire"
    ]
    
    print("\n=== CONVERSATION SIMULATION ===")
    for i, message in enumerate(test_messages, 1):
        print(f"\n👤 Client Message {i}: {message}")
        
        response = orchestrator.process_client_message(message)
        
        if response and 'orchestrator_response' in response:
            print(f"🤖 Assistant: {response['orchestrator_response']}")
            print(f"📊 Phase: {response.get('current_phase', 'unknown')} → {response.get('next_phase', 'unknown')}")
            print(f"📈 Completeness: {response.get('conversation_completeness', {}).get('overall', 0):.1%}")
            
            if response.get('architectural_brief'):
                print("\n📋 ARCHITECTURAL BRIEF GENERATED!")
                brief = response['architectural_brief']
                print(f"Project: {brief['project_summary']}")
        else:
            print("❌ Error: No valid response received")
