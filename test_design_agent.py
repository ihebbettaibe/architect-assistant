#!/usr/bin/env python3
"""
Test script for the Design Agent
Demonstrates the conversational flow and technical options integration
"""

import sys
import os
import json

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.design_agent import DesignAgent

def test_design_conversation():
    """Test a complete design conversation flow"""
    
    print("🎨 Testing Design Agent - Architecture Assistant")
    print("=" * 60)
    
    # Initialize the design agent
    agent = DesignAgent()
    
    # Simulate a conversation flow
    conversation_messages = [
        "Bonjour, je voudrais créer un intérieur chaleureux et élégant pour ma future maison.",
        "J'aime beaucoup les styles contemporains avec des touches de chaleur. Je préfère les espaces lumineux et ouverts.",
        "Pour le sol, je penche plutôt vers le marbre, et j'aimerais avoir un faux plafond pour créer une belle ambiance.",
        "Oui, le marbre Thala Beige me plaît beaucoup. Pour la cuisine, je souhaite qu'elle soit équipée. Et pour la sécurité, une porte blindée serait préférable.",
        "Pour les salles de bain, je préfère de la qualité importée. Et j'aimerais une pré-installation pour la climatisation."
    ]
    
    conversation_history = []
    
    for i, message in enumerate(conversation_messages):
        print(f"\n--- Étape {i+1} ---")
        print(f"👤 Client: {message}")
        
        # Process the message
        response = agent.process_message(message, conversation_history)
        
        print(f"🤖 Agent Design: {response['text']}")
        print(f"📊 Stage: {response['stage']}")
        
        if response.get('extracted_info'):
            print(f"🔍 Informations extraites:")
            print(json.dumps(response['extracted_info'], indent=2, ensure_ascii=False))
        
        # Add to conversation history
        conversation_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response['text']}
        ])
        
        # If we have design preferences, show them
        if response.get('design_preferences'):
            print(f"\n✅ PRÉFÉRENCES FINALES:")
            print(json.dumps(response['design_preferences'], indent=2, ensure_ascii=False))
        
        print("-" * 60)
    
    # Generate final design brief
    print(f"\n🎯 BRIEF DE DESIGN FINAL:")
    design_brief = agent.generate_design_brief()
    print(json.dumps(design_brief, indent=2, ensure_ascii=False))

def test_technical_options():
    """Test the technical options dictionary"""
    
    print("\n🔧 TESTING TECHNICAL OPTIONS")
    print("=" * 60)
    
    agent = DesignAgent()
    
    print("Options techniques disponibles:")
    print(json.dumps(agent.technical_options, indent=2, ensure_ascii=False))

def test_architectural_styles():
    """Test architectural style recommendations"""
    
    print("\n🏛️ TESTING ARCHITECTURAL STYLES")
    print("=" * 60)
    
    agent = DesignAgent()
    
    print("Styles architecturaux disponibles:")
    for style, info in agent.architectural_styles.items():
        print(f"\n{style.upper()}:")
        print(f"  Caractéristiques: {', '.join(info['key_features'])}")
        print(f"  Contextes: {', '.join(info['suitable_contexts'])}")
        print(f"  Matériaux: {', '.join(info['typical_materials'])}")

def test_conversation_stages():
    """Test different conversation stages"""
    
    print("\n📝 TESTING CONVERSATION STAGES")
    print("=" * 60)
    
    agent = DesignAgent()
    
    # Test different types of messages
    test_messages = [
        ("Initial inquiry", "Je cherche des idées pour décorer ma nouvelle maison"),
        ("Style preference", "J'aime le style moderne et minimaliste"),
        ("Technical question", "Quel type de sol recommandez-vous pour un salon?"),
        ("Clarification", "Je ne suis pas sûr pour le faux plafond, quels sont les avantages?"),
    ]
    
    for stage_name, message in test_messages:
        print(f"\n{stage_name}:")
        print(f"Message: {message}")
        
        response = agent.process_message(message)
        print(f"Réponse: {response['text'][:200]}...")
        print(f"Stage détecté: {response['stage']}")

if __name__ == "__main__":
    print("🚀 DÉMONSTRATION DU DESIGN AGENT")
    print("=" * 80)
    
    # Run all tests
    test_technical_options()
    test_architectural_styles()
    test_conversation_stages()
    test_design_conversation()
    
    print("\n✅ TESTS TERMINÉS!")
    print("Le Design Agent est prêt à être intégré dans l'Architecture Assistant!")
