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
        """Generate targeted questions that are fully personalized based on conversation context"""
        try:
            # Get conversation context from session state if available
            conversation_context = []
            import streamlit as st
            if hasattr(st, "session_state") and "conversation_context" in st.session_state:
                conversation_context = st.session_state.conversation_context
            
            # Extract information from budget_info
            budget = budget_info.get("budget")
            budget_range = budget_info.get("budget_range")
            flexibility = budget_info.get("flexibility")
            financing = budget_info.get("financing")
            
            # Generate a customized prompt based on gathered information
            context_prompt = "L'utilisateur"
            
            if budget:
                context_prompt += f" a mentionné un budget de {budget} DT"
            elif budget_range:
                context_prompt += f" a mentionné un budget entre {budget_range[0]} et {budget_range[1]} DT"
            else:
                context_prompt += " n'a pas précisé de budget clair"
                
            if flexibility and flexibility != "unknown":
                context_prompt += f" avec une {flexibility} flexibilité"
                
            if financing and financing != "unknown":
                context_prompt += f" et un financement par {financing}"
            
            # Try to use the LLM to generate personalized follow-up questions
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """Vous êtes un spécialiste immobilier qui analyse les conversations et suggère des questions de suivi très personnalisées.
                        Générez exactement 4 questions de suivi personnalisées en français, basées sur les derniers échanges de la conversation.
                        Les questions doivent:
                        1. Être formulées comme des questions directes que l'utilisateur pourrait sélectionner et poser
                        2. Être spécifiques au contexte de la conversation immobilière
                        3. Porter principalement sur les terrains constructibles en Tunisie
                        4. Aider à affiner la recherche immobilière ou mieux comprendre les besoins
                        5. Ne pas répéter des informations déjà données
                        
                        Répondez uniquement avec les 4 questions, une par ligne, sans numérotation ni autre texte."""
                    }
                ] + [msg for msg in conversation_context[-5:] if isinstance(msg, dict)],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=256
            )
            
            # Extract questions from response
            custom_questions = response.choices[0].message.content.strip().split("\n")
            
            # Clean up questions - remove any numbering or bullet points
            clean_questions = []
            for q in custom_questions:
                q = q.strip()
                # Remove leading numbers, dashes, asterisks, etc.
                q = q.lstrip("0123456789.- •*")
                q = q.strip()
                if q and len(q) > 10:  # Ensure it's a real question with decent length
                    clean_questions.append(q)
            
            # Ensure we have exactly 4 questions
            while len(clean_questions) < 4:
                # Fallback questions based on what we know
                fallbacks = []
                
                if not budget and not budget_range:
                    fallbacks.append("Quel budget envisagez-vous pour votre projet immobilier ?")
                elif budget:
                    fallbacks.append(f"Seriez-vous prêt à ajuster votre budget de {budget} DT si nécessaire ?")
                
                # Location questions
                fallbacks.append("Quelles zones géographiques vous intéressent particulièrement ?")
                fallbacks.append("Préférez-vous un terrain proche du centre-ville ou dans une zone périphérique ?")
                
                # Project questions
                fallbacks.append("Avez-vous des exigences particulières concernant la forme ou la topographie du terrain ?")
                fallbacks.append("Quelles sont vos priorités: proximité des services, vue, accessibilité ?")
                fallbacks.append("Avez-vous déjà un projet architectural en tête pour ce terrain ?")
                
                # Add unique fallback questions until we have 4
                for q in fallbacks:
                    if q not in clean_questions:
                        clean_questions.append(q)
                        if len(clean_questions) >= 4:
                            break
            
            return clean_questions[:4]  # Return exactly 4 questions
            
        except Exception as e:
            # Fallback questions if API call fails
            print(f"Error generating custom questions: {e}")
            return [
                "Pouvez-vous préciser davantage votre budget pour ce projet immobilier ?",
                "Quelle ville ou région vous intéresse particulièrement ?",
                "Quelle superficie minimale recherchez-vous pour votre terrain ?",
                "Avez-vous des contraintes particulières pour votre projet immobilier ?"
            ]
    
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