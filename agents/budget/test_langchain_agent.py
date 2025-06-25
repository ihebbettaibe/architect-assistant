#!/usr/bin/env python3
"""
Test the LangChain agent with CouchDB
"""

import sys
import os
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_budget_agent import create_langchain_budget_agent

def test_langchain_agent():
    """Test the LangChain agent functionality"""
    
    # Set the API key directly
    os.environ['GROQ_API_KEY'] = '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'
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("💡 Please set your Groq API key in environment or .env file")
        return
    
    try:
        print("🤖 Initializing LangChain budget agent...")
        agent = create_langchain_budget_agent(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            use_couchdb=True
        )
        print("✅ Agent initialized successfully")
        
        # Test terrain search in Sousse
        print("\n💬 Testing terrain search in Sousse...")
        test_message = "Je cherche un terrain à Sousse pour 200k DT"
        
        response = agent.chat(test_message)
        
        print(f"📋 Agent response:")
        print(f"   Response: {response.get('response', 'No response')}")
        print(f"   Properties found: {len(response.get('properties', []))}")
        print(f"   Context: {response.get('context', {})}")
        
        if response.get('properties'):
            print("\n🏠 Sample terrain properties:")
            for i, prop in enumerate(response['properties'][:5], 1):
                print(f"   {i}. {prop.get('prix', 'N/A')} DT, {prop.get('surface', 'N/A')} m², {prop.get('ville', 'N/A')}")
        
        # Test a second query with different parameters
        print("\n💬 Testing general budget query...")
        test_message2 = "Je cherche une propriété avec un budget de 300000 DT à Bizerte"
        
        response2 = agent.chat(test_message2)
        
        print(f"📋 Agent response 2:")
        print(f"   Response: {response2.get('response', 'No response')}")
        print(f"   Properties found: {len(response2.get('properties', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LangChain Budget Agent with CouchDB")
    print("=" * 50)
    test_langchain_agent()
    print("=" * 50)
