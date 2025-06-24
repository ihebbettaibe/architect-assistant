from typing import Dict, Any, List
import json
import pandas as pd
import numpy as np

class ClientInterface:
    def process_client_input(self, client_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the Budget Agent - processes client input and returns structured output
        
        Args:
            client_input: Natural language input from client about budget/financial constraints
            conversation_context: Context from previous conversation or other agents
            
        Returns:
            Structured JSON output with budget analysis and questions
        """
        # Extract budget information from client input
        budget_info = self._extract_budget_from_text(client_input, conversation_context)
        
        # Generate targeted questions
        targeted_questions = self._generate_targeted_questions(budget_info, conversation_context)
        
        # Detect inconsistencies
        inconsistencies = self._detect_budget_inconsistencies(budget_info, conversation_context)
        
        # Provide budget suggestions
        suggestions = self._generate_budget_suggestions(budget_info, conversation_context)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(budget_info, inconsistencies)
        
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "timestamp": pd.Timestamp.now().isoformat(),
            "budget_analysis": {
                "extracted_budget": budget_info.get("budget", None),
                "budget_range": budget_info.get("budget_range", None),
                "budget_flexibility": budget_info.get("flexibility", "unknown"),
                "financing_status": budget_info.get("financing", "unknown")
            },
            "inconsistencies_detected": inconsistencies,
            "targeted_questions": targeted_questions,
            "suggestions": suggestions,
            "reliability_score": reliability_score,
            "confidence_level": self._get_confidence_level(reliability_score),
            "next_actions": self._suggest_next_actions(budget_info, reliability_score)
        }
    
    def _extract_budget_from_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract budget information from natural language text"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an expert at extracting budget and financial information from client conversations.
                        Extract the following information and return as JSON:
                        {
                            "budget": explicit budget amount in DT (null if not mentioned),
                            "budget_range": [min, max] if range mentioned,
                            "flexibility": "strict/flexible/negotiable/unknown",
                            "financing": "cash/loan/mixed/unknown",
                            "timeline": "urgent/normal/flexible/unknown",
                            "budget_confidence": "certain/approximate/unsure"
                        }
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Extract budget information from: {text}"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Budget extraction error: {e}")
            return {"budget": None, "flexibility": "unknown"}
    
    def _generate_targeted_questions(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """Generate targeted questions to refine budget understanding"""
        questions = []
        
        if not budget_info.get("budget") and not budget_info.get("budget_range"):
            questions.append("Avez-vous une idée du budget que vous souhaitez consacrer à ce projet ?")
            questions.append("Quel montant maximum seriez-vous prêt(e) à investir ?")
        
        if budget_info.get("financing") == "unknown":
            questions.append("Avez-vous déjà une solution de financement (fonds propres, crédit, mixte) ?")
        
        if budget_info.get("flexibility") == "unknown":
            questions.append("Votre budget est-il strict ou avez-vous une marge de manœuvre ?")
        
        if budget_info.get("timeline") == "unknown":
            questions.append("Quel est votre délai souhaité pour la réalisation du projet ?")
        
        # Add contextual questions based on project type
        if context and context.get("project_type"):
            project_type = context["project_type"]
            if project_type == "construction":
                questions.append("Avez-vous déjà le terrain ou est-il inclus dans le budget ?")
            elif project_type == "renovation":
                questions.append("Quel pourcentage du budget souhaitez-vous allouer aux gros œuvre vs finitions ?")
        
        return questions[:3]  # Return max 3 most relevant questions
    
    def _detect_budget_inconsistencies(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Detect potential inconsistencies in budget information"""
        inconsistencies = []
        budget = budget_info.get("budget", 0)
        
        if context:
            # Check budget vs project type
            project_type = context.get("project_type")
            surface = context.get("surface", 0)
            
            if budget and surface and project_type:
                budget_per_m2 = budget / surface
                
                # Tunisian construction cost benchmarks (rough estimates)
                cost_benchmarks = {
                    "construction": {"min": 800, "max": 2000},  # DT/m²
                    "renovation": {"min": 400, "max": 1200},
                    "extension": {"min": 600, "max": 1500}
                }
                
                if project_type in cost_benchmarks:
                    benchmark = cost_benchmarks[project_type]
                    if budget_per_m2 < benchmark["min"]:
                        inconsistencies.append({
                            "type": "budget_too_low",
                            "message": f"Le budget semble insuffisant pour {project_type} ({budget_per_m2:.0f} DT/m²). Coût typique: {benchmark['min']}-{benchmark['max']} DT/m²",
                            "severity": "high"
                        })
                    elif budget_per_m2 > benchmark["max"] * 1.5:
                        inconsistencies.append({
                            "type": "budget_very_high",
                            "message": f"Le budget semble très élevé pour {project_type} ({budget_per_m2:.0f} DT/m²). Vérifiez si cela inclut des éléments spéciaux.",
                            "severity": "medium"
                        })
        
        return inconsistencies
    
    def _generate_budget_suggestions(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """Generate budget optimization suggestions"""
        suggestions = []
        budget = budget_info.get("budget", 0)
        
        if budget_info.get("flexibility") == "strict" and budget:
            suggestions.append("Avec un budget strict, je recommande de prévoir une marge de 10-15% pour les imprévus.")
        
        if budget_info.get("financing") == "loan":
            suggestions.append("Pour un financement par crédit, pensez à obtenir une pré-approbation avant de finaliser le projet.")
        
        if context and context.get("surface") and budget:
            surface = context["surface"]
            budget_per_m2 = budget / surface
            if budget_per_m2 < 1000:
                suggestions.append("Pour optimiser le budget, considérez une approche par phases ou des matériaux locaux.")
        
        suggestions.append("Je peux vous connecter avec des propriétés similaires pour valider votre estimation.")
        
        return suggestions
    
    def _calculate_reliability_score(self, budget_info: Dict[str, Any], inconsistencies: List[Dict]) -> float:
        """Calculate reliability score for the budget analysis"""
        score = 1.0
        
        # Reduce score for missing information
        if not budget_info.get("budget") and not budget_info.get("budget_range"):
            score -= 0.4
        
        if budget_info.get("budget_confidence") == "unsure":
            score -= 0.2
        elif budget_info.get("budget_confidence") == "approximate":
            score -= 0.1
        
        # Reduce score for inconsistencies
        for inconsistency in inconsistencies:
            if inconsistency["severity"] == "high":
                score -= 0.3
            elif inconsistency["severity"] == "medium":
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_confidence_level(self, reliability_score: float) -> str:
        """Convert reliability score to confidence level"""
        if reliability_score >= 0.8:
            return "high"
        elif reliability_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _suggest_next_actions(self, budget_info: Dict[str, Any], reliability_score: float) -> List[str]:
        """Suggest next actions based on budget analysis"""
        actions = []
        
        if reliability_score < 0.6:
            actions.append("collect_more_budget_info")
            actions.append("clarify_financing_options")
        
        if budget_info.get("budget"):
            actions.append("search_comparable_properties")
            actions.append("validate_budget_with_market_data")
        
        if reliability_score >= 0.7:
            actions.append("proceed_to_style_preferences")
            actions.append("assess_regulatory_constraints")
        
        return actions