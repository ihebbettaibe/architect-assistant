from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json

class RealEstateState(TypedDict):
    """State for the real estate consultation workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_data: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    current_step: str
    consultation_complete: bool

class RealEstateOrchestraAgent:
    """
    LangGraph-based Orchestra Agent for Real Estate Development in Tunisia
    Implements the decision tree workflow using LangGraph nodes and edges
    """
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(RealEstateState)
        
        # Add nodes for each decision point
        workflow.add_node("start_consultation", self.start_consultation_node)
        workflow.add_node("terrain_ownership", self.terrain_ownership_node)
        workflow.add_node("recommend_lotissements", self.recommend_lotissements_node)
        workflow.add_node("zone_type", self.zone_type_node)
        workflow.add_node("agricultural_zone", self.agricultural_zone_node)
        workflow.add_node("lotissement_approval", self.lotissement_approval_node)
        workflow.add_node("recommend_approved_lotissements", self.recommend_approved_lotissements_node)
        workflow.add_node("terrain_type", self.terrain_type_node)
        workflow.add_node("customization_hub", self.customization_hub_node)
        workflow.add_node("false_ceiling", self.false_ceiling_node)
        workflow.add_node("entrance_door", self.entrance_door_node)
        workflow.add_node("exterior_joinery", self.exterior_joinery_node)
        workflow.add_node("shutters", self.shutters_node)
        workflow.add_node("flooring", self.flooring_node)
        workflow.add_node("kitchen", self.kitchen_node)
        workflow.add_node("bathroom", self.bathroom_node)
        workflow.add_node("air_conditioning", self.air_conditioning_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # Set entry point
        workflow.set_entry_point("start_consultation")
        
        # Add edges based on the decision tree logic
        workflow.add_edge("start_consultation", "terrain_ownership")
        workflow.add_conditional_edges(
            "terrain_ownership",
            self._route_terrain_ownership,
            {
                "no_terrain": "recommend_lotissements",
                "has_terrain": "zone_type"
            }
        )
        
        workflow.add_conditional_edges(
            "zone_type",
            self._route_zone_type,
            {
                "agricultural": "agricultural_zone",
                "constructible": "lotissement_approval"
            }
        )
        
        workflow.add_conditional_edges(
            "lotissement_approval",
            self._route_lotissement_approval,
            {
                "not_approved": "recommend_approved_lotissements",
                "approved": "terrain_type"
            }
        )
        
        workflow.add_edge("terrain_type", "customization_hub")
        workflow.add_edge("customization_hub", "false_ceiling")
        workflow.add_edge("false_ceiling", "entrance_door")
        workflow.add_edge("entrance_door", "exterior_joinery")
        workflow.add_edge("exterior_joinery", "shutters")
        workflow.add_edge("shutters", "flooring")
        workflow.add_edge("flooring", "kitchen")
        workflow.add_edge("kitchen", "bathroom")
        workflow.add_edge("bathroom", "air_conditioning")
        workflow.add_edge("air_conditioning", "generate_report")
        
        # Terminal nodes
        workflow.add_edge("recommend_lotissements", "generate_report")
        workflow.add_edge("agricultural_zone", "generate_report")
        workflow.add_edge("recommend_approved_lotissements", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def start_consultation_node(self, state: RealEstateState) -> RealEstateState:
        """Initialize the consultation"""
        welcome_message = """🏠 Bienvenue dans votre consultation immobilière personnalisée
═══════════════════════════════════════════════════════════════════
Je vais vous guider à travers les différentes options pour votre projet immobilier en Tunisie."""
        
        state["messages"].append(AIMessage(content=welcome_message))
        state["user_data"] = {}
        state["recommendations"] = []
        state["current_step"] = "terrain_ownership"
        state["consultation_complete"] = False
        
        return state
    
    def terrain_ownership_node(self, state: RealEstateState) -> RealEstateState:
        """Ask about terrain ownership"""
        question = """📍 ÉTAPE 1: Possession du terrain
        
Possédez-vous un terrain ?
1. Oui
2. Non

Veuillez répondre par 1 ou 2."""
        
        state["messages"].append(AIMessage(content=question))
        state["current_step"] = "terrain_ownership"
        
        # In a real implementation, you would wait for user input here
        # For demo purposes, we'll simulate user input
        return state
    
    def _route_terrain_ownership(self, state: RealEstateState) -> str:
        """Route based on terrain ownership"""
        # In practice, this would check the last user message
        # For demo, we'll use user_data if available
        has_terrain = state["user_data"].get("has_terrain", False)
        return "has_terrain" if has_terrain else "no_terrain"
    
    def recommend_lotissements_node(self, state: RealEstateState) -> RealEstateState:
        """Recommend lotissements for users without terrain"""
        recommendation = """🏘️ RECOMMANDATION PERSONNALISÉE:
        
Nous proposons des lotissements sur tout le territoire tunisien en partenariat 
avec des agences spécialisées (Tecnocasa, etc.).

✅ Avantages:
- Couverture nationale
- Partenaires certifiés
- Accompagnement complet
- Choix variés selon vos besoins"""
        
        state["messages"].append(AIMessage(content=recommendation))
        state["recommendations"].append({
            "type": "lotissement_partnership",
            "description": "Lotissements en partenariat avec agences",
            "coverage": "Tout le territoire tunisien",
            "partners": ["Tecnocasa"]
        })
        
        return state
    
    def zone_type_node(self, state: RealEstateState) -> RealEstateState:
        """Ask about zone type"""
        question = """🌍 ÉTAPE 2: Type de zone
        
Dans quel type de zone se trouve votre terrain ?
1. Zone Agricole
2. Zone Constructible + Lotissement

Veuillez répondre par 1 ou 2."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def _route_zone_type(self, state: RealEstateState) -> str:
        """Route based on zone type"""
        zone_type = state["user_data"].get("zone_type", "constructible")
        return "agricultural" if zone_type == "agricole" else "constructible"
    
    def agricultural_zone_node(self, state: RealEstateState) -> RealEstateState:
        """Handle agricultural zone"""
        message = """🌾 ZONE AGRICOLE IDENTIFIÉE
        
Votre terrain se trouve en zone agricole. Nous allons adapter notre proposition 
en conséquence.

Veuillez indiquer la superficie du terrain réservé au projet (en m²):"""
        
        state["messages"].append(AIMessage(content=message))
        state["recommendations"].append({
            "type": "agricultural_development",
            "description": "Développement adapté pour zone agricole",
            "requirements": ["Respect des normes agricoles", "Surface minimale requise"]
        })
        
        return state
    
    def lotissement_approval_node(self, state: RealEstateState) -> RealEstateState:
        """Ask about lotissement approval"""
        question = """📋 ÉTAPE 3: Statut du lotissement
        
Le lotissement est-il approuvé ?
1. Oui, il est approuvé
2. Non, pas encore approuvé

Veuillez répondre par 1 ou 2."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def _route_lotissement_approval(self, state: RealEstateState) -> str:
        """Route based on lotissement approval"""
        approved = state["user_data"].get("lotissement_approved", False)
        return "approved" if approved else "not_approved"
    
    def recommend_approved_lotissements_node(self, state: RealEstateState) -> RealEstateState:
        """Recommend approved lotissements"""
        recommendation = """✅ LOTISSEMENTS APPROUVÉS DISPONIBLES
        
Nous proposons des lotissements déjà approuvés à la vente.

🏆 Avantages:
- Procédures administratives simplifiées
- Démarrage immédiat possible
- Sécurité juridique garantie
- Accompagnement dans le choix"""
        
        state["messages"].append(AIMessage(content=recommendation))
        state["recommendations"].append({
            "type": "approved_lotissements",
            "description": "Lotissements pré-approuvés disponibles",
            "benefits": ["Procédures simplifiées", "Démarrage rapide", "Sécurité juridique"]
        })
        
        return state
    
    def terrain_type_node(self, state: RealEstateState) -> RealEstateState:
        """Ask about terrain configuration"""
        question = """🏗️ ÉTAPE 4: Configuration du terrain
        
Quel type de construction souhaitez-vous ?
1. Villas en Bande Continue
2. Villas Jumelées  
3. Villas à Implantation Isolée

Chaque option a ses avantages spécifiques. Veuillez répondre par 1, 2 ou 3."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def customization_hub_node(self, state: RealEstateState) -> RealEstateState:
        """Central hub for customization options"""
        message = """🎨 ÉTAPE 5: Configuration & Personnalisation
        
Excellent ! Maintenant personnalisons votre villa selon vos préférences.
Nous allons passer en revue chaque élément pour créer votre maison idéale.

⚡ Processus de personnalisation démarré..."""
        
        state["messages"].append(AIMessage(content=message))
        return state
    
    def false_ceiling_node(self, state: RealEstateState) -> RealEstateState:
        """Handle false ceiling choice"""
        question = """🏠 Faux plafond
        
Souhaitez-vous installer un faux plafond ?
- Améliore l'isolation phonique et thermique
- Permet l'intégration de l'éclairage LED
- Coût calculé selon la surface

1. Oui, avec faux plafond
2. Non, plafond traditionnel"""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def entrance_door_node(self, state: RealEstateState) -> RealEstateState:
        """Handle entrance door choice"""
        question = """🚪 Porte d'entrée
        
Quel type de porte d'entrée préférez-vous ?
1. Porte blindée (sécurité renforcée)
2. Porte en bois plein (style traditionnel)

La porte blindée offre une sécurité supérieure."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def exterior_joinery_node(self, state: RealEstateState) -> RealEstateState:
        """Handle exterior joinery choice"""
        question = """🪟 Menuiserie extérieure
        
Choisissez le matériau pour vos fenêtres et portes extérieures:
1. Aluminium - TPR (durabilité, design moderne)
2. PVC - Wintech (isolation, prix attractif)

Chaque option a ses avantages spécifiques."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def shutters_node(self, state: RealEstateState) -> RealEstateState:
        """Handle shutters choice"""
        question = """🪟 Stores pour le RDC
        
Quel style de stores souhaitez-vous ?
1. Type Extrudé sans fer forgé (moderne, épuré)
2. Type Injecté avec fer forgé (traditionnel, sécurisé)

Le fer forgé ajoute une touche esthétique et sécuritaire."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def flooring_node(self, state: RealEstateState) -> RealEstateState:
        """Handle flooring choice"""
        question = """🏠 Revêtement de sol
        
Choisissez votre revêtement de sol:

1. GRÈS
   - SOMOCER (Tunisien, rapport qualité/prix)
   - Espagnol (Importé, haute qualité)

2. MARBRE  
   - Thala beige ou gris (Tunisien, authentique)
   - Italien (Importé, prestige)

Veuillez répondre par 1 ou 2."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def kitchen_node(self, state: RealEstateState) -> RealEstateState:
        """Handle kitchen choice"""
        question = """🍳 Cuisine équipée
        
Souhaitez-vous une cuisine entièrement équipée ?

1. Oui - Cuisine équipée complète
   - Fournisseurs: DELTA CUISINE / CUISINA
   - Électroménager inclus
   - Design personnalisé

2. Non - Cuisine à aménager selon vos goûts

La cuisine équipée vous fait gagner du temps et garantit la cohérence."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def bathroom_node(self, state: RealEstateState) -> RealEstateState:
        """Handle bathroom specifications"""
        message = """🛁 Salle de bains - Équipements Premium
        
Pour votre salle de bains, nous proposons des équipements haut de gamme:

✅ SANITAIRES: Import Allemagne (qualité supérieure)
✅ ROBINETTERIE: 
   - Option Allemagne (prestige)
   - Option SOPAL Tunisie (qualité locale)

Ces équipements sont inclus dans notre offre standard."""
        
        state["messages"].append(AIMessage(content=message))
        state["recommendations"].append({
            "option": "bathroom_premium",
            "sanitaires": "Import Allemagne",
            "robinetterie": "Allemagne ou SOPAL Tunisie"
        })
        return state
    
    def air_conditioning_node(self, state: RealEstateState) -> RealEstateState:
        """Handle air conditioning choice"""
        question = """❄️ Climatisation
        
Souhaitez-vous installer la climatisation ?
1. Oui - Installation complète
2. Non - Pré-installation uniquement (câblage et évacuations)

La pré-installation vous permet d'ajouter la climatisation plus tard 
à moindre coût."""
        
        state["messages"].append(AIMessage(content=question))
        return state
    
    def generate_report_node(self, state: RealEstateState) -> RealEstateState:
        """Generate final consultation report"""
        report = """📋 RAPPORT DE CONSULTATION PERSONNALISÉE
═══════════════════════════════════════════════════════════════════

🎯 RÉSUMÉ DE VOTRE PROJET:
"""
        
        # Add user data summary
        if state["user_data"]:
            report += "\n📊 VOS PRÉFÉRENCES:\n"
            for key, value in state["user_data"].items():
                report += f"• {key}: {value}\n"
        
        # Add recommendations
        if state["recommendations"]:
            report += "\n🏆 NOS RECOMMANDATIONS:\n"
            for i, rec in enumerate(state["recommendations"], 1):
                report += f"{i}. {rec.get('description', 'Recommandation personnalisée')}\n"
        
        report += """
📞 PROCHAINES ÉTAPES:
• Validation de votre projet avec notre équipe
• Établissement du devis détaillé  
• Planning de réalisation
• Suivi personnalisé

✨ Merci de votre confiance ! Notre équipe vous contactera sous 48h."""
        
        state["messages"].append(AIMessage(content=report))
        state["consultation_complete"] = True
        
        return state
    
    def run_consultation(self, user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete consultation workflow"""
        initial_state = {
            "messages": [],
            "user_data": user_inputs or {},
            "recommendations": [],
            "current_step": "start",
            "consultation_complete": False
        }
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "messages": [msg.content for msg in final_state["messages"]],
            "recommendations": final_state["recommendations"],
            "user_data": final_state["user_data"],
            "consultation_complete": final_state["consultation_complete"]
        }

# Usage example
if __name__ == "__main__":
    # Initialize the agent
    agent = RealEstateOrchestraAgent()
    
    # Example user data (in practice, this would come from user interactions)
    sample_user_data = {
        "has_terrain": True,
        "zone_type": "constructible", 
        "lotissement_approved": True,
        "terrain_type": "isole",
        "surface": "500"
    }
    
    # Run consultation
    result = agent.run_consultation(sample_user_data)
    
    # Display results
    print("=== CONSULTATION RESULTS ===")
    for message in result["messages"]:
        print(f"\n{message}")
        print("-" * 50)
    
    print(f"\n=== RECOMMENDATIONS SUMMARY ===")
    for rec in result["recommendations"]:
        print(f"• {rec}")