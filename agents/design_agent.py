import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class DesignPreferences(BaseModel):
    """Schema for design preferences"""
    mots_clés_style: List[str] = Field(..., description="Mots-clés décrivant le style architectural souhaité")
    faux_plafond: bool = Field(..., description="Si le client veut un faux plafond dans toutes les pièces")
    porte_entree: str = Field(..., description="Type de porte d'entrée: 'blindée' ou 'bois plein'")
    menuiserie_ext: str = Field(..., description="Type de menuiserie extérieure: 'Aluminium TPR' ou 'PVC Wintech'")
    revêtement_sol: str = Field(..., description="Type de revêtement de sol (grès ou marbre) et sa provenance")
    cuisine_équipée: bool = Field(..., description="Si le client souhaite une cuisine équipée")
    salle_de_bain: Dict[str, str] = Field(..., description="Préférences pour les salles de bain (sanitaire et robinetterie)")
    climatisation: str = Field(..., description="Options de climatisation: 'non', 'pré-installation' ou 'installée'")
    notes_client: str = Field(..., description="Notes supplémentaires sur les préférences du client")

class DesignAgent:
    """
    Agent Design - Specialized AI agent for architectural design and style recommendations
    
    Objective: Guide clients through design choices and technical options,
    collect preferences, and generate structured output.
    
    Outputs: Design preferences in structured JSON format
    """
    
    def __init__(self, design_db_path: str = "knowledge_base/design"):
        """Initialize the Design Agent with design knowledge base"""
        self.agent_name = "Agent Design"
        self.agent_role = "Design Style and Technical Options Specialist"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Simple conversation history instead of deprecated ConversationBufferMemory
        self.conversation_history = []
        
        # Initialize design knowledge base if it exists
        self.design_kb_path = design_db_path
        self.design_kb = self._initialize_design_kb() if os.path.exists(design_db_path) else None
        
        # Dictionary of architectural styles with key features
        self.architectural_styles = {
            "contemporary": {
                "key_features": ["clean lines", "open floor plans", "minimalist", "sustainable materials"],
                "suitable_contexts": ["urban", "suburban", "coastal"],
                "typical_materials": ["concrete", "glass", "steel", "sustainable composites"]
            },
            "mediterranean": {
                "key_features": ["stucco walls", "low-pitched tile roofs", "arches", "warm colors"],
                "suitable_contexts": ["coastal", "suburban", "rural"],
                "typical_materials": ["terracotta", "stucco", "wrought iron", "ceramic tiles"]
            },
            "modern": {
                "key_features": ["asymmetrical", "flat roofs", "large windows", "minimal ornamentation"],
                "suitable_contexts": ["urban", "suburban"],
                "typical_materials": ["concrete", "glass", "steel", "metal"]
            },
        }
        
        # Technical options reference for Tunisian architectural projects
        self.technical_options = {
            "faux_plafond": ["oui", "non"],
            "porte_entree": ["blindée", "bois plein"],
            "menuiserie_ext": ["Aluminium TPR", "PVC Wintech"],
            "stores_rdc": ["Extrudé (sans fer forgé)", "Injecté (avec fer forgé)"],
            "revêtement_sol": {
                "grès": ["Tunisien (SOMOCER)", "Importé (Espagnol)"],
                "marbre": ["Thala Beige ou Gris (Tunisien)", "Importé (Italie)"]
            },
            "cuisine_équipée": ["non", "oui (DELTA CUISINE ou CUISINA)"],
            "salle_de_bain": {
                "sanitaire": ["importé (Allemagne)"],
                "robinetterie": ["importé (Allemagne)", "Tunisie (SOPAL)"]
            },
            "climatisation": ["non", "pré-installation", "installée"]
        }
        
        # Core design principles for architectural projects
        self.design_principles = [
            "Functionality and livability",
            "Natural light optimization",
            "Spatial flow and circulation",
            "Material harmony and durability",
            "Energy efficiency and sustainability",
            "Cultural and contextual appropriateness",
            "Aesthetic coherence and style consistency"
        ]
    
    def _initialize_design_kb(self):
        """Initialize design knowledge base if available"""
        try:
            return Chroma(
                persist_directory=self.design_kb_path,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error initializing design knowledge base: {e}")
            return None
            
    def add_design_reference(self, reference_text: str, metadata: Dict[str, Any] = None):
        """Add a design reference to the knowledge base"""
        if not os.path.exists(self.design_kb_path):
            os.makedirs(self.design_kb_path, exist_ok=True)
            self.design_kb = Chroma(
                persist_directory=self.design_kb_path,
                embedding_function=self.embeddings
            )
        
        if self.design_kb:
            self.design_kb.add_texts(
                texts=[reference_text],
                metadatas=[metadata or {}]
            )
            
    def search_design_references(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search the design knowledge base for relevant references"""
        if not self.design_kb:
            return []
            
        results = self.design_kb.similarity_search_with_score(query, k=k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
        
    def get_style_recommendations(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get style recommendations based on client preferences"""
        # Extract key preferences
        keywords = preferences.get("mots_clés_style", [])
        query = " ".join(keywords)
        
        # If we have a knowledge base, search for relevant references
        if self.design_kb:
            references = self.search_design_references(query)
            if references:
                return references
        
        # Fall back to architectural styles dictionary
        matching_styles = []
        
        for style_name, style_info in self.architectural_styles.items():
            # Calculate a simple match score based on keyword overlap
            key_features = style_info.get("key_features", [])
            match_score = sum(1 for kw in keywords if any(kf in kw or kw in kf for kf in key_features))
            
            if match_score > 0:
                matching_styles.append({
                    "style": style_name,
                    "match_score": match_score,
                    "key_features": style_info.get("key_features", []),
                    "materials": style_info.get("typical_materials", [])
                })
                
        # Sort by match score
        return sorted(matching_styles, key=lambda x: x["match_score"], reverse=True)
        
    def visualize_style_preferences(self, preferences: Dict[str, Any]):
        """Generate a visualization of style preferences"""
        # Implementation would depend on available visualization tools
        # Placeholder for future implementation
        pass
    
    def analyze_design_preferences(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user input to extract design preferences and style indicators
        
        Args:
            user_input: User message containing design-related content
            
        Returns:
            Dictionary with extracted design preferences
        """
        prompt = f"""
        Based on the following client message, extract any design preferences, 
        architectural style indicators, material preferences, or aesthetic values.
        
        Client message: "{user_input}"
        
        Respond with a JSON object containing:
        1. "identified_styles": List of architectural styles mentioned or implied
        2. "material_preferences": List of materials mentioned or implied
        3. "spatial_preferences": Preferences about space use (open, compartmentalized, etc.)
        4. "aesthetic_values": Key aesthetic values (minimalist, ornate, natural, etc.)
        5. "confidence_score": Your confidence in this assessment (0-100)
        
        JSON response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from the response
            result = self._extract_json(result_text)
            return result
            
        except Exception as e:
            print(f"Error analyzing design preferences: {e}")
            return {
                "identified_styles": [],
                "material_preferences": [],
                "spatial_preferences": [],
                "aesthetic_values": [],
                "confidence_score": 0
            }
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text, handling potential formatting issues"""
        try:
            # Find anything that looks like a JSON object
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return {}
        except json.JSONDecodeError:
            # Fallback to empty dict if JSON parsing fails
            return {}
    
    def recommend_styles(self, preferences: Dict[str, Any], project_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend architectural styles based on preferences and project details
        
        Args:
            preferences: Extracted design preferences
            project_details: Details about the project (location, type, etc.)
            
        Returns:
            Dictionary with style recommendations and explanations
        """
        # Construct context for the recommendation
        context = {
            "preferences": preferences,
            "project": project_details
        }
        
        # Convert context to a prompt
        context_str = json.dumps(context, ensure_ascii=False)
        
        prompt = f"""
        You are an expert architectural design consultant. Based on the following client 
        preferences and project details, recommend the top 3 most suitable architectural styles.
        
        Client context:
        {context_str}
        
        For each recommended style:
        1. Explain why it's suitable for this client
        2. List key design elements that should be incorporated
        3. Suggest adaptations that might be needed for the specific project
        4. Rate the match on a scale of 0-100
        
        Respond with a JSON object containing:
        1. "primary_recommendation": Object with style name, explanation, elements, adaptations, and match_score
        2. "secondary_recommendations": Array of objects with the same structure
        3. "design_principles": List of 3-5 key design principles that should guide this project
        
        JSON response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from the response
            result = self._extract_json(result_text)
            return result
            
        except Exception as e:
            print(f"Error generating style recommendations: {e}")
            return {
                "primary_recommendation": {
                    "style": "contemporary",
                    "explanation": "Error generating recommendations",
                    "elements": [],
                    "adaptations": [],
                    "match_score": 0
                },
                "secondary_recommendations": [],
                "design_principles": self.design_principles[:3]
            }
    
    def visualize_style_comparison(self, styles: List[str]) -> Dict[str, Any]:
        """
        Create a comparison of architectural styles
        
        Args:
            styles: List of architectural styles to compare
            
        Returns:
            Dictionary with comparison details
        """
        comparison = {}
        
        for style in styles:
            if style.lower() in self.architectural_styles:
                comparison[style] = self.architectural_styles[style.lower()]
        
        return {
            "style_comparison": comparison,
            "design_principles": self.design_principles
        }
    
    def process_design_query(self, query: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a design-related query and provide a comprehensive response
        
        Args:
            query: User's design question or request
            conversation_context: Current conversation context
            
        Returns:
            Response with design recommendations, explanations, and next steps
        """
        # Extract project details from context
        project_details = conversation_context.get("project_details", {})
        
        # Analyze design preferences in the query
        preferences = self.analyze_design_preferences(query)
        
        # If we already have style preferences in the context, merge them
        existing_preferences = conversation_context.get("style_preferences", {})
        if existing_preferences:
            for key in preferences:
                if key in existing_preferences and isinstance(preferences[key], list):
                    # Merge lists while removing duplicates
                    combined = list(set(existing_preferences[key] + preferences[key]))
                    preferences[key] = combined
        
        # Generate style recommendations
        recommendations = self.recommend_styles(preferences, project_details)
        
        # Get style comparisons if multiple styles are recommended
        # Handle cases where the JSON structure might be incomplete
        recommended_styles = []
        
        # Try to get primary recommendation style
        primary_rec = recommendations.get("primary_recommendation", {})
        if isinstance(primary_rec, dict) and "style" in primary_rec:
            recommended_styles.append(primary_rec["style"])
        else:
            # Fallback to a default style
            recommended_styles.append("contemporary")
        
        # Try to get secondary recommendations
        secondary_recs = recommendations.get("secondary_recommendations", [])
        for rec in secondary_recs:
            if isinstance(rec, dict) and "style" in rec:
                recommended_styles.append(rec["style"])
            
        style_comparison = self.visualize_style_comparison(recommended_styles)
        
        # Prepare the response
        response = {
            "preferences": preferences,
            "recommendations": recommendations,
            "style_comparison": style_comparison,
            "design_principles": recommendations.get("design_principles", self.design_principles[:3])
        }
        
        return response
        
    def generate_design_explanation(self, design_data: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of design recommendations
        
        Args:
            design_data: Data from process_design_query
            
        Returns:
            String with formatted explanation
        """
        prompt = f"""
        Convert the following architectural design data into a clear, helpful explanation for a client.
        Make it conversational, educational, and visually descriptive.
        
        Design data:
        {json.dumps(design_data, ensure_ascii=False)}
        
        Your explanation should:
        1. Acknowledge the client's preferences
        2. Explain the primary style recommendation with vivid details
        3. Briefly mention alternative styles
        4. Explain key design principles that will guide the project
        5. End with a question that helps move the conversation forward
        
        Response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating design explanation: {e}")
            return "I apologize, but I encountered an error while generating your design explanation. Let's try a different approach to discuss your design preferences."
    
    def process_message(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a message from the client about design preferences.
        
        Args:
            message: The client's message
            conversation_history: Previous conversation history
            
        Returns:
            A dict containing the response and any extracted design preferences
        """
        # Update conversation history if provided
        if conversation_history:
            self.conversation_history = conversation_history.copy()
        
        # Add the current message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Determine conversation stage based on memory
        conversation_stage = self._determine_conversation_stage()
        
        if conversation_stage == "initial":
            response = self._handle_initial_inquiry(message)
        elif conversation_stage == "style_exploration":
            response = self._handle_style_exploration(message)
        elif conversation_stage == "technical_options":
            response = self._handle_technical_options(message)
        elif conversation_stage == "clarification":
            response = self._handle_clarification(message)
        elif conversation_stage == "finalization":
            response = self._handle_finalization(message)
        else:
            response = self._handle_generic_inquiry(message)
            
        # Add the response to history
        self.conversation_history.append({"role": "assistant", "content": response["text"]})
        
        return response
    
    def _determine_conversation_stage(self) -> str:
        """
        Determine the current stage of the conversation based on history.
        
        Returns:
            A string representing the current conversation stage
        """
        # If this is the first message, we're in the initial stage
        if len(self.conversation_history) <= 1:
            return "initial"
            
        # Check if we have enough context for technical options
        style_keywords = ["style", "préférence", "esthétique", "ambiance", "goût", "design"]
        technical_keywords = ["technique", "matériau", "équipement", "plafond", "sol", "menuiserie", "porte", "cuisine", "salle de bain"]
        
        # Count mentions of different types of keywords
        style_mentions = 0
        technical_mentions = 0
        
        for msg in self.conversation_history:
            content = msg["content"].lower()
            style_mentions += sum(1 for keyword in style_keywords if keyword in content)
            technical_mentions += sum(1 for keyword in technical_keywords if keyword in content)
        
        # Determine stage based on keyword counts and conversation length
        if len(self.conversation_history) >= 10 or technical_mentions > 5:
            return "finalization"
        elif technical_mentions > style_mentions:
            return "technical_options"
        elif style_mentions > 0:
            return "style_exploration"
        else:
            return "initial"
    
    def _handle_initial_inquiry(self, message: str) -> Dict[str, Any]:
        """
        Handle the initial inquiry from the client.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text and any extracted information
        """
        prompt = f"""
        Tu es un expert en architecture et design intérieur. Un client vient de te contacter avec le message suivant:
        "{message}"
        
        Commence une conversation pour explorer ses préférences de style architectural. 
        Pose des questions ouvertes sur le style, l'ambiance et le type d'espace qu'il souhaite.
        Sois chaleureux et professionnel. Ne mentionne pas encore les options techniques spécifiques.
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "text": response.choices[0].message.content,
            "extracted_info": {},
            "stage": "initial"
        }
    
    def _handle_style_exploration(self, message: str) -> Dict[str, Any]:
        """
        Handle style exploration, starting to introduce some technical options.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text and any extracted style preferences
        """
        # Get previous conversation context
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        prompt = f"""
        Tu es un expert en architecture et design intérieur. Voici la conversation jusqu'à présent:
        
        {conversation_context}
        
        Le client vient de dire: "{message}"
        
        Basé sur ce que tu as appris de ses préférences de style, commence à introduire naturellement 
        1 ou 2 options techniques spécifiques qui correspondent à ses goûts esthétiques.
        
        Par exemple, tu peux parler du revêtement de sol ou des faux plafonds si cela s'intègre 
        naturellement dans la conversation. Propose des choix spécifiques, mais reste concentré 
        principalement sur les aspects esthétiques et le style global.
        
        Options techniques à considérer:
        - Faux plafond dans toutes les pièces: oui / non
        - Revêtement sol: Grès (Tunisien/Espagnol) ou Marbre (Tunisien/Importé)
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract style keywords if possible
        extraction_prompt = f"""
        À partir de la conversation suivante, identifie les mots-clés qui décrivent 
        le style architectural souhaité par le client. Renvoie uniquement une liste de mots-clés 
        (3 à 5 mots maximum) au format JSON.
        
        Conversation:
        {conversation_context}
        
        Format de sortie attendu:
        {{
            "mots_clés_style": ["mot1", "mot2", "mot3"]
        }}
        """
        
        try:
            extraction_response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=200,
                temperature=0.2
            )
            
            extracted_json = json.loads(extraction_response.choices[0].message.content)
        except:
            extracted_json = {"mots_clés_style": []}
        
        return {
            "text": response.choices[0].message.content,
            "extracted_info": extracted_json,
            "stage": "style_exploration"
        }
    
    def _handle_technical_options(self, message: str) -> Dict[str, Any]:
        """
        Handle technical options discussions.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text and extracted technical preferences
        """
        # Get previous conversation context
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        # Determine which technical options to focus on based on conversation
        focus_options = self._determine_focus_options(conversation_context)
        
        prompt = f"""
        Tu es un expert en architecture et design intérieur. Voici la conversation jusqu'à présent:
        
        {conversation_context}
        
        Le client vient de dire: "{message}"
        
        Maintenant, discute plus en profondeur des options techniques suivantes, en présentant les 
        choix disponibles de manière naturelle, et en expliquant brièvement les avantages de chaque option:
        
        {focus_options}
        
        Intègre ces options dans ta réponse de manière conversationnelle, et pose des questions pour 
        comprendre les préférences du client sur ces aspects techniques précis.
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        # Try to extract technical preferences if possible
        extraction_prompt = f"""
        À partir de la conversation suivante, extrais les préférences techniques du client.
        Renvoie uniquement un objet JSON avec les préférences identifiées.
        Si une préférence n'est pas mentionnée, ne l'inclus pas dans le JSON.
        
        Conversation:
        {conversation_context}
        
        Client: "{message}"
        
        Options techniques possibles:
        {json.dumps(self.technical_options, ensure_ascii=False, indent=2)}
        
        Format de sortie attendu (exemple, n'inclure que les options mentionnées):
        {{
            "faux_plafond": true,
            "porte_entree": "blindée",
            "revêtement_sol": "Marbre - Thala Beige"
        }}
        """
        
        try:
            extraction_response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=300,
                temperature=0.2
            )
            
            extracted_json = json.loads(extraction_response.choices[0].message.content)
        except:
            extracted_json = {}
        
        return {
            "text": response.choices[0].message.content,
            "extracted_info": extracted_json,
            "stage": "technical_options"
        }
    
    def _determine_focus_options(self, conversation_context: str) -> str:
        """
        Determine which technical options to focus on based on conversation context.
        
        Args:
            conversation_context: The full conversation history
            
        Returns:
            A string containing the technical options to focus on
        """
        # Check which options have been discussed
        options_discussed = {}
        for option, _ in self.technical_options.items():
            if option in conversation_context.lower():
                options_discussed[option] = True
        
        # Select 2-3 options to focus on that haven't been discussed or need more detail
        focus_options = []
        
        # First priority: core options that haven't been discussed
        core_options = ["revêtement_sol", "faux_plafond", "menuiserie_ext"]
        for option in core_options:
            if option not in options_discussed:
                focus_options.append(option)
                if len(focus_options) >= 2:
                    break
        
        # Second priority: other options that haven't been discussed
        if len(focus_options) < 2:
            for option in self.technical_options:
                if option not in options_discussed and option not in focus_options:
                    focus_options.append(option)
                    if len(focus_options) >= 2:
                        break
        
        # Format the focus options for the prompt
        formatted_options = []
        for option in focus_options:
            if option == "revêtement_sol":
                formatted_options.append(f"Revêtement de sol: Grès (Tunisien SOMOCER / Importé Espagnol) ou Marbre (Thala Beige/Gris ou Importé d'Italie)")
            elif option == "faux_plafond":
                formatted_options.append(f"Faux plafond dans toutes les pièces: oui / non")
            elif option == "porte_entree":
                formatted_options.append(f"Porte d'entrée: blindée / bois plein")
            elif option == "menuiserie_ext":
                formatted_options.append(f"Menuiserie extérieure: Aluminium TPR / PVC Wintech")
            elif option == "stores_rdc":
                formatted_options.append(f"Stores RDC: Extrudé (sans fer forgé) / Injecté (avec fer forgé)")
            elif option == "cuisine_équipée":
                formatted_options.append(f"Cuisine équipée: non / oui (DELTA CUISINE ou CUISINA)")
            elif option == "salle_de_bain":
                formatted_options.append(f"Salles de bains: Appareils sanitaires importés (Allemagne) et Robinetterie importée (Allemagne) ou Tunisienne (SOPAL)")
            elif option == "climatisation":
                formatted_options.append(f"Climatisation: non / pré-installation / installée")
                
        return "\n".join(formatted_options)
    
    def _handle_clarification(self, message: str) -> Dict[str, Any]:
        """
        Handle clarification questions when information is ambiguous.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text and any clarified information
        """
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        # Identify what needs clarification
        clarification_prompt = f"""
        À partir de la conversation suivante et identifie les aspects qui nécessitent une clarification 
        concernant les préférences de design du client:
        
        {conversation_context}
        
        Le client vient de dire: "{message}"
        
        Quelle(s) information(s) essentielle(s) manque(nt) ou sont ambiguës concernant:
        1. Le style architectural général
        2. Les préférences de matériaux
        3. Les options techniques spécifiques
        
        Renvoie uniquement un JSON listant les points à clarifier.
        """
        
        try:
            clarification_response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": clarification_prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            clarification_points = json.loads(clarification_response.choices[0].message.content)
        except:
            clarification_points = {"points_à_clarifier": ["style général", "préférences techniques"]}
        
        # Generate response with clarification questions
        response_prompt = f"""
        Tu es un expert en architecture et design intérieur. Voici la conversation jusqu'à présent:
        
        {conversation_context}
        
        Le client vient de dire: "{message}"
        
        D'après mon analyse, j'ai besoin de clarifier les points suivants:
        {json.dumps(clarification_points, ensure_ascii=False, indent=2)}
        
        Formule des questions précises mais conversationnelles pour clarifier ces points.
        Évite de poser plus de 2 questions. Sois courtois et professionnel.
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": response_prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "text": response.choices[0].message.content,
            "extracted_info": {"clarification_points": clarification_points},
            "stage": "clarification"
        }
    
    def _handle_finalization(self, message: str) -> Dict[str, Any]:
        """
        Handle finalization and generate structured output.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text and the finalized design preferences
        """
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        # Generate structured summary
        summary_prompt = f"""
        À partir de la conversation suivante, génère un résumé structuré des préférences de design du client.
        Assure-toi de remplir tous les champs du JSON, en faisant des déductions raisonnables si nécessaire.
        
        Conversation complète:
        {conversation_context}
        
        Dernier message du client: "{message}"
        
        Renvoie un JSON avec exactement cette structure:
        {{
          "mots_clés_style": ["mot1", "mot2", "mot3"],
          "faux_plafond": true/false,
          "porte_entree": "blindée"/"bois plein",
          "menuiserie_ext": "Aluminium TPR"/"PVC Wintech",
          "revêtement_sol": "Type - Provenance",
          "cuisine_équipée": true/false,
          "salle_de_bain": {{
            "sanitaire": "importé - Allemagne",
            "robinetterie": "Tunisie - SOPAL"/"importé - Allemagne"
          }},
          "climatisation": "non"/"pré-installation"/"installée",
          "notes_client": "Résumé des préférences et besoins spécifiques."
        }}
        
        Ne laisse aucun champ vide. Si l'information n'est pas explicite dans la conversation, déduis-la
        à partir du contexte et des préférences générales du client. Le JSON doit être parfaitement formaté.
        """
        
        try:
            summary_response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=600,
                temperature=0.4
            )
            
            design_preferences = json.loads(summary_response.choices[0].message.content)
        except Exception as e:
            print(f"Error parsing JSON summary: {e}")
            design_preferences = {
                "mots_clés_style": ["moderne", "élégant"],
                "faux_plafond": True,
                "porte_entree": "blindée",
                "menuiserie_ext": "Aluminium TPR",
                "revêtement_sol": "Grès - Importé (Espagnol)",
                "cuisine_équipée": True,
                "salle_de_bain": {
                    "sanitaire": "importé - Allemagne",
                    "robinetterie": "importé - Allemagne"
                },
                "climatisation": "pré-installation",
                "notes_client": "Information insuffisante pour détails complets."
            }
        
        # Generate a final response
        final_response_prompt = f"""
        Tu es un expert en architecture et design intérieur. Voici la conversation complète avec le client:
        
        {conversation_context}
        
        Dernier message du client: "{message}"
        
        J'ai synthétisé ses préférences de design. Rédige un message final qui:
        1. Résume les principales préférences de style et options techniques choisies
        2. Confirme que ces informations seront utilisées pour la suite du projet
        3. Explique que ces choix pourront être affinés plus tard avec l'architecte
        4. Demande si le client a des questions sur les options sélectionnées
        
        Sois chaleureux, professionnel et concis.
        """
        
        final_response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": final_response_prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "text": final_response.choices[0].message.content,
            "extracted_info": design_preferences,
            "stage": "finalization",
            "design_preferences": design_preferences
        }
    
    def _handle_generic_inquiry(self, message: str) -> Dict[str, Any]:
        """
        Handle general inquiries that don't fit into specific stages.
        
        Args:
            message: The client's message
            
        Returns:
            A dict containing the response text
        """
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-5:]])
        
        prompt = f"""
        Tu es un expert en architecture et design intérieur. Voici les derniers échanges de la conversation:
        
        {conversation_context}
        
        Le client vient de dire: "{message}"
        
        Réponds de manière informative et professionnelle. Si c'est pertinent, oriente la conversation 
        vers les préférences de style ou les options techniques pour un projet architectural.
        """
        
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "text": response.choices[0].message.content,
            "extracted_info": {},
            "stage": "generic"
        }
    
    def generate_design_brief(self) -> Dict[str, Any]:
        """
        Generate a comprehensive design brief based on the conversation history.
        
        Returns:
            A dict containing the complete design brief
        """
        conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        prompt = f"""
        À partir de la conversation suivante, génère un brief de design architectural complet.
        
        Conversation complète:
        {conversation_context}
        
        Le brief doit inclure:
        1. Un résumé des préférences stylistiques (mots-clés, ambiance générale)
        2. Toutes les options techniques choisies
        3. Des recommandations complémentaires basées sur les choix du client
        4. Des notes sur des points à clarifier ou à développer
        
        Renvoie un objet JSON structuré avec tous ces éléments.
        """
        
        try:
            brief_response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4
            )
            
            design_brief = json.loads(brief_response.choices[0].message.content)
        except:
            design_brief = {
                "style": {
                    "mots_clés": ["moderne", "élégant", "fonctionnel"],
                    "ambiance": "Contemporain avec touches chaleureuses"
                },
                "options_techniques": {
                    "faux_plafond": True,
                    "porte_entree": "blindée",
                    "menuiserie_ext": "Aluminium TPR",
                    "revêtement_sol": "Grès - Importé (Espagnol)",
                    "cuisine_équipée": True,
                    "salle_de_bain": {
                        "sanitaire": "importé - Allemagne",
                        "robinetterie": "importé - Allemagne"
                    },
                    "climatisation": "pré-installation"
                },
                "recommandations": [
                    "Considérer un éclairage indirect pour amplifier l'effet du faux plafond",
                    "Envisager des matériaux durables pour les zones à fort passage"
                ],
                "points_à_clarifier": [
                    "Couleurs préférées pour les finitions",
                    "Besoins spécifiques pour les espaces extérieurs"
                ]
            }
        
        return design_brief
