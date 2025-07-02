# 🎨 Design Agent - Architecture Assistant

## Vue d'ensemble

L'Agent Design est un composant intelligent du système Architecture Assistant qui guide les clients à travers une discussion fluide pour comprendre leurs préférences architecturales, choix esthétiques et configurations techniques pour leur futur logement.

## Fonctionnalités Principales

### 🎯 Objectifs
- Collecter les préférences de style architectural du client
- Proposer des options techniques adaptées au contexte tunisien
- Générer un brief de design structuré en format JSON
- Guider la conversation de manière naturelle et professionnelle

### 🗣️ Flux de Conversation Intelligent
L'agent suit un flux conversationnel adaptatif avec plusieurs phases :

1. **Initial** : Questions ouvertes sur le style et l'ambiance
2. **Exploration du Style** : Approfondissement des préférences esthétiques
3. **Options Techniques** : Présentation des choix techniques spécifiques
4. **Clarification** : Questions de précision si nécessaire
5. **Finalisation** : Génération du brief structuré

### 🔧 Options Techniques Intégrées

L'agent propose les options techniques suivantes, adaptées au marché tunisien :

- **Faux plafond** : oui / non
- **Porte d'entrée** : blindée / bois plein  
- **Menuiserie extérieure** : Aluminium TPR / PVC Wintech
- **Stores RDC** : Extrudé (sans fer forgé) / Injecté (avec fer forgé)
- **Revêtement sol** :
  - Grès : Tunisien (SOMOCER) / Importé (Espagnol)
  - Marbre : Thala Beige ou Gris (Tunisien) / Importé (Italie)
- **Cuisine équipée** : non / oui (DELTA CUISINE ou CUISINA)
- **Salles de bains** :
  - Appareils sanitaires : importé (Allemagne)
  - Robinetterie : importé (Allemagne) ou Tunisie (SOPAL)
- **Climatisation** : non / pré-installation / installée

## Installation et Utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

### Utilisation Standalone
```python
from agents.design_agent import DesignAgent

agent = DesignAgent()
response = agent.process_message("Je veux un intérieur moderne et chaleureux")
print(response['text'])
```

### Intégration avec l'Orchestrateur
L'agent est automatiquement intégré dans le système d'orchestration principal :

```python
from agents.orchestrator_agent import ArchitectureAssistantOrchestrator

orchestrator = ArchitectureAssistantOrchestrator()
# L'agent design est disponible via orchestrator.agents["design"]
```

### Interface Streamlit
Lancez l'interface dédiée :
```bash
streamlit run streamlit_design_app.py
```

## Exemple de Conversation

```
👤 Client: "Je voudrais un intérieur chaleureux et élégant pour ma nouvelle maison."

🤖 Agent: "Excellente idée ! Pour créer cette ambiance chaleureuse et élégante, 
j'aimerais mieux comprendre vos goûts. Préférez-vous un style plutôt contemporain 
avec des lignes épurées, ou êtes-vous attiré par quelque chose de plus traditionnel 
avec des éléments décoratifs ? Et concernant les matériaux, avez-vous une préférence 
entre le marbre et le grès pour le revêtement de sol ?"

👤 Client: "J'aime le contemporain, et je penche pour le marbre."

🤖 Agent: "Parfait ! Le style contemporain se marie magnifiquement avec le marbre. 
Le marbre Thala Beige tunisien apporterait cette élégance naturelle que vous 
recherchez, ou préféreriez-vous explorer les marbres importés d'Italie ? 
Et pour amplifier cette ambiance raffinée, avez-vous pensé à un faux plafond 
dans les pièces principales ?"
```

## Format de Sortie

L'agent génère un JSON structuré avec les préférences collectées :

```json
{
  "mots_clés_style": ["contemporain", "élégant", "chaleureux"],
  "faux_plafond": true,
  "porte_entree": "blindée",
  "menuiserie_ext": "Aluminium TPR",
  "revêtement_sol": "Marbre - Thala Beige",
  "cuisine_équipée": true,
  "salle_de_bain": {
    "sanitaire": "importé - Allemagne",
    "robinetterie": "Tunisie - SOPAL"
  },
  "climatisation": "pré-installation",
  "notes_client": "Préfère les espaces ouverts avec beaucoup de lumière naturelle"
}
```

## Tests

### Exécuter tous les tests
```bash
python test_design_integration.py
```

### Test de l'agent seul
```bash
python test_design_agent.py
```

### Interface de test
```bash
streamlit run streamlit_design_app.py
```

## Architecture Technique

### Composants Principaux
- **DesignAgent** : Classe principale de l'agent
- **DesignPreferences** : Modèle Pydantic pour la validation des données
- **ConversationBufferMemory** : Gestion de l'historique de conversation
- **Groq LLM** : Modèle de langage pour la génération de réponses

### Intégration LangChain
L'agent utilise LangChain pour :
- Gestion de la mémoire conversationnelle
- Parsing des réponses structurées
- Intégration avec les modèles de langage

### Base de Connaissances
- Styles architecturaux prédéfinis
- Options techniques du marché tunisien
- Principes de design architectural

## Évolutions Futures

- [ ] Intégration d'images et références visuelles
- [ ] Recommandations basées sur le budget
- [ ] Génération de moodboards automatiques
- [ ] Intégration avec des catalogues fournisseurs
- [ ] Support multilingue (Arabe/Français/Anglais)

## Support

Pour toute question ou support technique, référez-vous à la documentation principale du projet Architecture Assistant.
