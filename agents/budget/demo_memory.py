#!/usr/bin/env python3
"""
Demonstration of conversation memory and URL functionality
"""

from langchain_budget_agent import create_langchain_budget_agent

def demo_conversation_memory():
    """Demonstrate conversation memory functionality"""
    try:
        agent = create_langchain_budget_agent()
        
        print("🧠 Demonstration de la mémoire conversationnelle")
        print("=" * 60)
        
        # First interaction - establish budget and location
        print("\n💬 Première interaction:")
        print("User: J'ai un budget de 300000 DT pour un appartement à Sousse")
        response1 = agent.invoke({"input": "J'ai un budget de 300000 DT pour un appartement à Sousse"})
        print(f"Assistant: {response1['output'][:150]}...")
        
        # Second interaction - ask for URLs without repeating budget/location
        print("\n💬 Deuxième interaction (test de mémoire):")
        print("User: Donne-moi les URLs pour ces propriétés")
        response2 = agent.invoke({"input": "Donne-moi les URLs pour ces propriétés"})
        
        # Check if the agent remembered the context
        if "url" in response2['output'].lower() and "sousse" in response2['output'].lower():
            print("✅ SUCCÈS: L'agent se souvient du budget et de la ville!")
            print(f"Assistant: {response2['output'][:200]}...")
        else:
            print("❌ ÉCHEC: L'agent n'a pas utilisé la mémoire correctement")
            print(f"Assistant: {response2['output'][:200]}...")
        
        # Third interaction - change budget and test memory update
        print("\n💬 Troisième interaction (changement de budget):")
        print("User: Finalement, mon budget est de 250000 DT")
        response3 = agent.invoke({"input": "Finalement, mon budget est de 250000 DT"})
        print(f"Assistant: {response3['output'][:150]}...")
        
        # Fourth interaction - test updated memory
        print("\n💬 Quatrième interaction (test mémoire mise à jour):")
        print("User: Cherche des propriétés maintenant")
        response4 = agent.invoke({"input": "Cherche des propriétés maintenant"})
        
        if "250" in response4['output'] or "250000" in response4['output']:
            print("✅ SUCCÈS: L'agent a mis à jour sa mémoire avec le nouveau budget!")
        else:
            print("⚠️  L'agent pourrait avoir des difficultés avec la mise à jour de mémoire")
        
        print(f"Assistant: {response4['output'][:200]}...")
        
        print("\n" + "=" * 60)
        print("✅ Démonstration de la mémoire conversationnelle terminée!")
        
    except Exception as e:
        print(f"❌ Erreur durant la démonstration: {e}")

if __name__ == "__main__":
    demo_conversation_memory()
