import streamlit as st
import sys
import os
import json
from datetime import datetime

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

try:
    from agents.design_agent import DesignAgent
except ImportError:
    st.error("❌ Could not import DesignAgent. Please check the import paths.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="🎨 Design Agent - Architecture Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        border-color: #4169e1;
        color: #000080;
    }
    .agent-message {
        background: linear-gradient(135deg, #f0fff0, #e6ffe6);
        border-color: #32cd32;
        color: #006400;
    }
    .technical-options {
        background: linear-gradient(135deg, #fff8f0, #ffe6d9);
        border: 2px solid #ff8c00;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .design-brief {
        background: linear-gradient(135deg, #f8f0ff, #f0e6ff);
        border: 2px solid #9370db;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stage-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stage-initial { background-color: #e3f2fd; color: #1976d2; }
    .stage-style-exploration { background-color: #f3e5f5; color: #7b1fa2; }
    .stage-technical-options { background-color: #fff3e0; color: #f57c00; }
    .stage-clarification { background-color: #e8f5e8; color: #388e3c; }
    .stage-finalization { background-color: #fce4ec; color: #c2185b; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'design_agent' not in st.session_state:
    with st.spinner("🎨 Initialisation de l'Agent Design avec LangChain..."):
        try:
            st.session_state.design_agent = DesignAgent()
            st.session_state.initialized = True
            st.success("✅ Agent Design initialisé avec succès!")
            st.info("🔧 Fonctionnalités LangChain actives : JSON parsing robuste, gestion mémoire avancée")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            st.session_state.initialized = False
            st.info("💡 Vérifiez que votre clé API GROQ est configurée dans le fichier .env")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'design_preferences' not in st.session_state:
    st.session_state.design_preferences = {}

# Main title
st.markdown('<h1 class="main-header">🎨 Agent Design - Architecture Assistant</h1>', unsafe_allow_html=True)

# Sidebar for agent status and options
with st.sidebar:
    st.header("🤖 Statut de l'Agent")
    
    if st.session_state.get('initialized', False):
        st.success("✅ Agent Design Prêt")
        
        # Show LangChain status
        agent = st.session_state.design_agent
        with st.expander("🔧 Statut LangChain"):
            st.write(f"**LLM Principal:** {agent.llm.model_name}")
            st.write(f"**LLM Analyse:** {agent.analysis_llm.model_name}")
            st.write(f"**Mémoire:** {'✅ Active' if agent.memory else '❌ Désactivée'}")
            st.write(f"**Templates:** {len(agent.prompt_templates)} disponibles")
            
    else:
        st.error("❌ Agent Non Initialisé")
    
    st.header("🎯 Options Techniques")
    
    # Show technical options available
    if st.session_state.get('initialized', False):
        with st.expander("📋 Options Disponibles"):
            st.json(st.session_state.design_agent.technical_options)
    
    st.header("🏛️ Styles Architecturaux")
    
    # Show architectural styles
    if st.session_state.get('initialized', False):
        with st.expander("🎨 Styles Disponibles"):
            styles_info = {}
            for style, info in st.session_state.design_agent.architectural_styles.items():
                styles_info[style] = {
                    "features": info["key_features"],
                    "contexts": info["suitable_contexts"]
                }
            st.json(styles_info)
    
    st.header("🔧 Actions")
    
    if st.button("🔄 Nouvelle Conversation", help="Effacer l'historique de conversation"):
        st.session_state.conversation_history = []
        st.session_state.design_preferences = {}
        if st.session_state.get('design_agent') and hasattr(st.session_state.design_agent, 'clear_memory'):
            st.session_state.design_agent.clear_memory()
        st.success("🧹 Conversation réinitialisée!")
        st.rerun()
    
    if st.button("📊 Générer Brief", help="Générer le brief de design final"):
        if st.session_state.conversation_history:
            brief = st.session_state.design_agent.generate_design_brief()
            st.session_state.design_brief = brief

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Conversation avec l'Agent Design")
    
    # Display conversation history
    if st.session_state.conversation_history:
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Vous:</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                stage = msg.get("stage", "generic")
                stage_class = f"stage-{stage.replace('_', '-')}"
                
                st.markdown(f"""
                <div class="chat-message agent-message">
                    <div class="stage-indicator {stage_class}">
                        📍 Phase: {stage.replace('_', ' ').title()}
                    </div>
                    <strong>🤖 Agent Design:</strong><br>
                    {msg["content"]}
                    <br><small>⏱️ Traité en {msg.get('processing_time', 0):.2f}s</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show extracted info if available
                if msg.get("extracted_info"):
                    with st.expander("🔍 Informations Extraites (LangChain)"):
                        st.json(msg["extracted_info"])
                
                # Show memory summary if available
                if msg.get("memory_summary"):
                    with st.expander("🧠 Résumé Mémoire"):
                        st.text(msg["memory_summary"][:500] + "..." if len(msg["memory_summary"]) > 500 else msg["memory_summary"])
    
    # Chat input
    if st.session_state.get('initialized', False):
        user_input = st.chat_input("💭 Décrivez vos préférences de design...")
        
        if user_input:
            # Process the message
            with st.spinner("🤔 L'agent réfléchit avec LangChain..."):
                try:
                    # Add processing time measurement
                    import time
                    start_time = time.time()
                    
                    response = st.session_state.design_agent.process_message(
                        user_input, 
                        st.session_state.conversation_history
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Add messages to history
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response["text"],
                        "stage": response["stage"],
                        "extracted_info": response.get("extracted_info", {}),
                        "processing_time": processing_time,
                        "memory_summary": response.get("memory_summary", "")
                    })
                    
                    # Update design preferences if available
                    if response.get("design_preferences"):
                        st.session_state.design_preferences = response["design_preferences"]
                    
                    # Show success message with timing
                    st.success(f"✅ Traité en {processing_time:.2f}s")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
                    st.error("💡 Vérifiez votre connexion et votre clé API GROQ")

with col2:
    st.header("📋 Informations Collectées")
    
    # Show current design preferences
    if st.session_state.design_preferences:
        st.markdown('<div class="design-brief">', unsafe_allow_html=True)
        st.subheader("✅ Préférences Finalisées")
        
        prefs = st.session_state.design_preferences
        
        # Style keywords
        if prefs.get("mots_clés_style"):
            st.write("🎨 **Style:**", ", ".join(prefs["mots_clés_style"]))
        
        # Technical options
        technical_items = [
            ("🏠 Faux plafond", prefs.get("faux_plafond", "Non spécifié")),
            ("🚪 Porte d'entrée", prefs.get("porte_entree", "Non spécifié")),
            ("🪟 Menuiserie ext.", prefs.get("menuiserie_ext", "Non spécifié")),
            ("🏠 Revêtement sol", prefs.get("revêtement_sol", "Non spécifié")),
            ("🍳 Cuisine équipée", prefs.get("cuisine_équipée", "Non spécifié")),
            ("❄️ Climatisation", prefs.get("climatisation", "Non spécifié"))
        ]
        
        for label, value in technical_items:
            if value != "Non spécifié":
                st.write(f"{label}: **{value}**")
        
        # Bathroom details
        if prefs.get("salle_de_bain"):
            sdb = prefs["salle_de_bain"]
            st.write("🛁 **Salle de bain:**")
            if sdb.get("sanitaire"):
                st.write(f"  • Sanitaire: {sdb['sanitaire']}")
            if sdb.get("robinetterie"):
                st.write(f"  • Robinetterie: {sdb['robinetterie']}")
        
        # Notes
        if prefs.get("notes_client"):
            st.write("📝 **Notes:** " + prefs["notes_client"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show extracted info from last message
    if st.session_state.conversation_history:
        last_msg = st.session_state.conversation_history[-1]
        if last_msg["role"] == "assistant" and last_msg.get("extracted_info"):
            st.markdown('<div class="technical-options">', unsafe_allow_html=True)
            st.subheader("🔍 Dernières Informations")
            
            extracted = last_msg["extracted_info"]
            for key, value in extracted.items():
                if isinstance(value, bool):
                    value = "Oui" if value else "Non"
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show final design brief if generated
    if hasattr(st.session_state, 'design_brief'):
        st.markdown('<div class="design-brief">', unsafe_allow_html=True)
        st.subheader("🎯 Brief de Design Final")
        st.json(st.session_state.design_brief)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🎨 Agent Design - Architecture Assistant | 
    Système intelligent pour la collecte des préférences de design architectural<br>
    <strong>Enhanced with LangChain:</strong> JSON parsing robuste • Gestion mémoire avancée • Templates structurés
</div>
""", unsafe_allow_html=True)

# Display usage instructions
with st.expander("📖 Guide d'utilisation"):
    st.markdown("""
    ### 🎯 Comment utiliser l'Agent Design:
    
    1. **Démarrage**: Décrivez vos goûts et préférences de style
    2. **Exploration**: L'agent vous guidera à travers différentes options
    3. **Détails techniques**: Spécifiez vos choix pour les matériaux et équipements
    4. **Finalisation**: Obtenez un résumé complet de vos préférences
    
    ### 💡 Exemples de messages:
    - "Je veux un style moderne et minimaliste"
    - "J'hésite entre le marbre et le grès pour le sol"
    - "Faut-il des faux plafonds partout?"
    - "Je veux une cuisine équipée et une porte blindée"
    
    ### 🔧 Fonctionnalités LangChain:
    - **JSON parsing robuste**: Extraction fiable des préférences
    - **Mémoire conversationnelle**: Contexte maintenu sur toute la session
    - **Templates structurés**: Prompts optimisés pour chaque phase
    """)

# Example prompts
if not st.session_state.conversation_history:
    st.markdown("### 💡 Exemples de messages pour commencer:")
    
    example_prompts = [
        "Je veux un intérieur chaleureux et élégant pour ma nouvelle maison",
        "J'aime les styles contemporains avec beaucoup de lumière naturelle",
        "Pouvez-vous m'aider à choisir entre le marbre et le grès pour le sol?",
        "Je voudrais une cuisine équipée et une porte d'entrée sécurisée"
    ]
    
    cols = st.columns(2)
    for i, prompt in enumerate(example_prompts):
        with cols[i % 2]:
            if st.button(f"💭 {prompt}", key=f"example_{i}"):
                # Simulate clicking with this prompt
                st.session_state.example_prompt = prompt
                st.rerun()

# Handle example prompt
if hasattr(st.session_state, 'example_prompt'):
    # Process the example prompt
    with st.spinner("🤔 L'agent réfléchit..."):
        try:
            response = st.session_state.design_agent.process_message(
                st.session_state.example_prompt, 
                st.session_state.conversation_history
            )
            
            # Add messages to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": st.session_state.example_prompt
            })
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response["text"],
                "stage": response["stage"],
                "extracted_info": response.get("extracted_info", {})
            })
            
            # Clean up
            del st.session_state.example_prompt
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement: {str(e)}")
