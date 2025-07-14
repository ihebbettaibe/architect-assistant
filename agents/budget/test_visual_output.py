#!/usr/bin/env python3
"""
Quick test of enhanced visual output for property URLs
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from langchain_budget_agent import create_langchain_budget_agent
    
    print("🎨 Test de l'affichage visuel amélioré")
    print("=" * 60)
    
    agent = create_langchain_budget_agent()
    
    # Test specific URL request
    print("\n📋 Test: Demande d'URLs avec budget spécifique")
    print("Question: 'Donne-moi les URLs des appartements avec un budget de 200000 DT à Tunis'")
    print("-" * 60)
    
    response = agent.invoke({
        "input": "Donne-moi les URLs des appartements avec un budget de 200000 DT à Tunis"
    })
    
    print("\n🤖 RÉPONSE DE L'AGENT:")
    print("=" * 60)
    print(response['output'])
    print("=" * 60)
    
    print("\n✅ Test terminé! L'agent devrait maintenant afficher:")
    print("   • Nom clair de chaque propriété")
    print("   • Localisation précise (quartier, ville)")
    print("   • Prix et détails techniques")
    print("   • URL complète et cliquable")
    print("   • Formatage visuel avec émojis")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
