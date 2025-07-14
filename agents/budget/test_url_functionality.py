#!/usr/bin/env python3
"""
Test script to verify URL functionality in the enhanced budget agent
"""

from langchain_budget_agent import create_langchain_budget_agent

def test_url_requests():
    """Test different ways users might ask for URLs"""
    try:
        agent = create_langchain_budget_agent()
        
        test_queries = [
            "donne-moi l'URL pour cette propriété",
            "montre-moi les liens des maisons",
            "je veux voir les URLs des appartements avec mon budget de 200000 DT à Tunis",
            "cherche des propriétés avec URLs à Sousse budget 300000 DT",
            "recommande des maisons avec liens pour 250000 DT"
        ]
        
        print("🧪 Testing URL functionality...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 30)
            
            try:
                response = agent.invoke({"input": query})
                print(f"✅ Response: {response['output'][:200]}...")
                
                # Check if response mentions URLs
                if 'url' in response['output'].lower() or 'lien' in response['output'].lower():
                    print("🎯 ✓ Response includes URL information")
                else:
                    print("⚠️  Response might be missing URL information")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ URL functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")

if __name__ == "__main__":
    test_url_requests()
