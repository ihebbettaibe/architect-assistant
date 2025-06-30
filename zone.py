import json
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Core dependencies
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerrainClassification:
    """Structure for terrain classification result"""
    classification: str  # isolée, jumelée, bande continue
    confidence: float
    reasoning: str
    location_context: str

class TerrainRulesEngine:
    """Engine for processing structured JSON rules"""
    
    def __init__(self, rules_file: str = None):
        self.rules = self._get_default_rules()
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str):
        """Load terrain classification rules from JSON"""
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"Loaded rules from {rules_file}")
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
    
    def _get_default_rules(self) -> Dict:
        """Default rules for Tunisia terrain classification"""
        return {
            "terrain_types": {
                "isolée": {
                    "description": "Habitation individuelle isolée",
                    "surface_min": 200,
                    "distance_min": 3.0
                },
                "jumelée": {
                    "description": "Habitation jumelée (2 logements accolés)",
                    "surface_min": 150,
                    "distance_min": 3.0
                },
                "bande_continue": {
                    "description": "Habitations en bande continue",
                    "surface_min": 100,
                    "distance_facade": 5.0
                }
            },
            "zones": {
                "urbaine": ["isolée", "jumelée", "bande_continue"],
                "périurbaine": ["isolée", "jumelée"],
                "rurale": ["isolée"]
            },
            "locations": {
                "Sousse": {"zones_spéciales": ["Khezama", "Port El Kantaoui"]},
                "Tunis": {"zones_spéciales": ["Sidi Bou Said", "La Marsa"]}
            }
        }
    
    def classify_terrain(self, terrain_info: Dict) -> TerrainClassification:
        """Classify terrain based on rules"""
        surface = terrain_info.get('surface', 0)
        zone = terrain_info.get('zone', 'urbaine')
        location = terrain_info.get('location', '')
        
        # Get allowed types for zone
        allowed_types = self.rules['zones'].get(zone, ['isolée'])
        possible_types = []
        
        # Check each terrain type
        for terrain_type in allowed_types:
            rules = self.rules['terrain_types'].get(terrain_type, {})
            surface_min = rules.get('surface_min', 0)
            
            if surface >= surface_min:
                possible_types.append(terrain_type)
        
        # Determine classification
        if not possible_types:
            return TerrainClassification(
                classification="non_classifiable",
                confidence=0.0,
                reasoning="Surface insuffisante ou zone non compatible",
                location_context=location
            )
        elif len(possible_types) == 1:
            return TerrainClassification(
                classification=possible_types[0],
                confidence=0.9,
                reasoning=f"Classification unique basée sur surface {surface}m² et zone {zone}",
                location_context=location
            )
        else:
            # Multiple options - choose based on surface
            if surface > 300:
                classification = "isolée"
            elif surface > 200:
                classification = "jumelée"
            else:
                classification = "bande_continue"
            
            return TerrainClassification(
                classification=classification,
                confidence=0.7,
                reasoning=f"Plusieurs options possibles, sélection basée sur surface {surface}m²",
                location_context=location
            )

class DocumentRAG:
    """Simple RAG system for documents"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_documents(self, document_paths: List[str]):
        """Load and process documents"""
        documents = []
        
        for path in document_paths:
            try:
                if path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif path.endswith('.txt'):
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded: {path}")
                
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
        
        if documents:
            texts = self.text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Processed {len(texts)} document chunks")
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class TerrainRAGSystem:
    """Main RAG system - simplified version"""
    
    def __init__(self, rules_file: str = None):
        self.rules_engine = TerrainRulesEngine(rules_file)
        self.document_rag = DocumentRAG()
        self.llm = self._init_llama()
        
    def _init_llama(self):
        """Initialize local Llama model"""
        try:
            llm = Ollama(model="llama2", temperature=0.1)
            logger.info("✅ Llama model initialized")
            return llm
        except Exception as e:
            logger.warning(f"❌ Llama not available: {e}")
            return None
    
    def load_documents(self, document_paths: List[str]):
        """Load documents into RAG"""
        self.document_rag.load_documents(document_paths)
    
    def extract_terrain_info(self, query: str) -> Dict:
        """Extract info from user query"""
        terrain_info = {'surface': 0, 'zone': 'urbaine', 'location': ''}
        
        # Extract location
        tunisia_cities = [
            'Tunis', 'Sousse', 'Sfax', 'Kairouan', 'Bizerte', 'Gabès', 
            'Khezama', 'Port El Kantaoui', 'Sidi Bou Said', 'La Marsa'
        ]
        
        query_lower = query.lower()
        for city in tunisia_cities:
            if city.lower() in query_lower:
                terrain_info['location'] = city
                break
        
        # Extract surface
        surface_match = re.search(r'(\d+)\s*m[²2]', query_lower)
        if surface_match:
            terrain_info['surface'] = int(surface_match.group(1))
        
        # Extract zone
        if any(word in query_lower for word in ['rural', 'campagne']):
            terrain_info['zone'] = 'rurale'
        elif any(word in query_lower for word in ['périurbain', 'banlieue']):
            terrain_info['zone'] = 'périurbaine'
        
        return terrain_info
    
    def answer_query(self, query: str) -> Dict:
        """Main method to answer questions"""
        # Extract terrain info
        terrain_info = self.extract_terrain_info(query)
        
        # Get classification from rules
        classification = self.rules_engine.classify_terrain(terrain_info)
        
        # Search documents
        relevant_docs = self.document_rag.search_documents(query)
        
        # Generate answer
        if self.llm and relevant_docs:
            answer = self._generate_llm_answer(query, classification, relevant_docs)
        else:
            answer = self._generate_simple_answer(classification, terrain_info)
        
        return {
            'answer': answer,
            'classification': classification.classification,
            'confidence': classification.confidence,
            'location': terrain_info['location'],
            'surface': terrain_info['surface'],
            'zone': terrain_info['zone']
        }
    
    def _generate_llm_answer(self, query: str, classification: TerrainClassification, docs: List[Document]) -> str:
        """Generate answer using Llama"""
        # Prepare context
        context = f"""
Classification: {classification.classification}
Confiance: {classification.confidence}
Raisonnement: {classification.reasoning}
Localisation: {classification.location_context}

Documents pertinents:
"""
        for i, doc in enumerate(docs[:2]):
            context += f"Doc {i+1}: {doc.page_content[:200]}...\n"
        
        prompt = f"""Tu es un expert en urbanisme tunisien. Réponds à cette question en français:

Question: {query}

Contexte:
{context}

Donne une réponse claire et précise avec:
1. Le type de terrain autorisé
2. La justification
3. Les conseils pratiques

Réponse:"""
        
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._generate_simple_answer(classification, {})
    
    def _generate_simple_answer(self, classification: TerrainClassification, terrain_info: Dict) -> str:
        """Generate simple rule-based answer"""
        if classification.classification == "non_classifiable":
            return "❌ Ce terrain ne peut pas être classifié selon les règles actuelles."
        
        answer = f"""✅ **Classification: {classification.classification.upper()}**

📊 Confiance: {classification.confidence:.1%}
📝 Justification: {classification.reasoning}"""
        
        if classification.location_context:
            answer += f"\n📍 Localisation: {classification.location_context}"
        
        return answer

# Streamlit Interface
def main():
    st.set_page_config(
        page_title="Tunisia Terrain RAG",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Classification des Terrains - Tunisie")
    st.markdown("*Système RAG avec Llama local*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Documents")
        
        # Upload JSON rules
        rules_file = st.file_uploader("Règles JSON", type=['json'])
        
        # Upload documents
        uploaded_docs = st.file_uploader(
            "Documents PDF/TXT", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True
        )
        
        # Check Ollama status
        st.header("🦙 Status Llama")
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if 'llama2' in result.stdout:
                st.success("✅ Llama2 disponible")
            else:
                st.warning("⚠️ Installer: `ollama pull llama2`")
        except:
            st.error("❌ Ollama non installé")
            st.markdown("[Installer Ollama](https://ollama.ai)")
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        rules_path = None
        if rules_file:
            rules_path = f"temp_rules.json"
            with open(rules_path, "wb") as f:
                f.write(rules_file.getbuffer())
        
        st.session_state.rag_system = TerrainRAGSystem(rules_path)
        
        # Load documents
        if uploaded_docs:
            doc_paths = []
            for doc in uploaded_docs:
                doc_path = f"temp_{doc.name}"
                with open(doc_path, "wb") as f:
                    f.write(doc.getbuffer())
                doc_paths.append(doc_path)
            
            with st.spinner("📚 Traitement des documents..."):
                st.session_state.rag_system.load_documents(doc_paths)
            st.success(f"✅ {len(doc_paths)} documents chargés")
    
    # Main interface
    st.header("💬 Votre Question")
    
    # Example questions
    examples = [
        "Quel type d'implantation pour un terrain de 250m² à Khezama Sousse ?",
        "Puis-je construire une maison isolée sur 180m² ?",
        "Règles pour habitation jumelée en zone urbaine ?",
    ]
    
    selected_example = st.selectbox("Exemples:", [""] + examples)
    
    # Query input
    query = st.text_area(
        "Décrivez votre terrain:",
        value=selected_example,
        height=100,
        placeholder="Ex: Terrain de 200m² à Sousse, zone urbaine..."
    )
    
    if st.button("🔍 Analyser", type="primary"):
        if query:
            with st.spinner("🤔 Analyse en cours..."):
                result = st.session_state.rag_system.answer_query(query)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📝 Réponse")
                    st.markdown(result['answer'])
                
                with col2:
                    st.markdown("### 📊 Détails")
                    
                    # Classification
                    classification = result['classification']
                    if classification == "non_classifiable":
                        st.error("❌ Non classifiable")
                    else:
                        st.success(f"✅ {classification.upper()}")
                    
                    # Info extracted
                    st.metric("Confiance", f"{result['confidence']:.1%}")
                    
                    if result['location']:
                        st.info(f"📍 {result['location']}")
                    
                    if result['surface'] > 0:
                        st.info(f"📐 {result['surface']}m²")
                    
                    st.info(f"🏘️ Zone {result['zone']}")
        else:
            st.warning("Veuillez poser une question")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 🚀 Instructions rapides:
    1. **Installer Ollama**: `curl https://ollama.ai/install.sh | sh`
    2. **Télécharger Llama**: `ollama pull llama2`  
    3. **Télécharger vos documents** PDF/JSON dans la sidebar
    4. **Poser votre question** avec localisation et surface
    
    💡 **Astuce**: Mentionnez la surface (ex: 200m²) et la ville pour de meilleurs résultats!
    """)

if __name__ == "__main__":
    main()