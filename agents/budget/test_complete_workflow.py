#!/usr/bin/env python3
"""
Test the complete workflow: search for terrain in Sousse for 200k DT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from agents.budget.langchain_budget_agent import create_langchain_budget_agent

def test_terrain_search():
    print("🔍 Testing complete terrain search workflow...")
    
    try:
        # Try to get API key from environment
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("⚠️ GROQ_API_KEY not found in environment variables")
            print("💡 Testing core functionality without LangChain agent...")
            test_core_functionality()
            return
        
        # Create the agent
        print("🔄 Creating LangChain budget agent...")
        agent = create_langchain_budget_agent(groq_api_key)
        
        if not agent:
            print("❌ Failed to create agent")
            return
        
        print("✅ Agent created successfully")
        
        # Test the search query
        query = "I'm looking for terrain in Sousse for 200k DT"
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            response = agent.chat(query)
            print("\n📋 Agent Response:")
            print("=" * 50)
            print(response.get('response', 'No response found'))
            print("=" * 50)
            
            # Show properties found
            properties = response.get('properties', [])
            if properties:
                print(f"\n🏠 Properties found: {len(properties)}")
                for i, prop in enumerate(properties[:3]):
                    price = prop.get('Price', prop.get('price', 'N/A'))
                    surface = prop.get('Surface', prop.get('surface', 'N/A'))
                    location = prop.get('Location', prop.get('City', prop.get('location', 'N/A')))
                    print(f"  {i+1}. {location} - {price} DT - {surface} m²")
            
        except Exception as e:
            print(f"❌ Error during agent invocation: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def test_core_functionality():
    """Test the core budget analysis functionality"""
    print("🔍 Testing core budget analysis functionality...")
    
    try:
        from budget_agent_base import EnhancedBudgetAgent
        
        # Create agent
        agent = EnhancedBudgetAgent(use_couchdb=True)
        print("✅ Core budget agent created successfully")
        
        # Test analysis
        result = agent.analyze_budget(
            city="Sousse",
            budget=200000,
            min_size=150,
            property_type="terrain"
        )
        
        print("\n📋 Analysis Result:")
        print("=" * 50)
        if result:
            print(f"Total properties found: {result.get('total_properties', 0)}")
            print(f"Within budget: {result.get('properties_in_budget', 0)}")
            print(f"Average price: {result.get('average_price', 0):.0f} DT")
            print(f"Average surface: {result.get('average_surface', 0):.0f} m²")
            
            properties = result.get('sample_properties', [])
            if properties:
                print(f"\nSample properties ({len(properties)}):")
                for i, prop in enumerate(properties[:3]):
                    price = prop.get('Price', prop.get('price', 'N/A'))
                    surface = prop.get('Surface', prop.get('surface', 'N/A'))
                    location = prop.get('Location', prop.get('City', prop.get('location', 'N/A')))
                    print(f"  {i+1}. {location} - {price} DT - {surface} m²")
        else:
            print("No results returned")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during core functionality test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_terrain_search()
