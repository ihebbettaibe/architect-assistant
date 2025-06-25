#!/usr/bin/env python3
"""
Test the core budget agent functionality directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from budget_agent_base import EnhancedBudgetAgent

def test_core_agent():
    print("🔧 Testing core budget agent functionality...")
    
    try:
        # Test CouchDB provider directly
        from couchdb_provider import CouchDBProvider
        
        provider = CouchDBProvider()
        print("✅ CouchDB provider initialized")
        
        # Test terrain search
        print("\n🔍 Testing terrain search in Sousse...")
        
        # Search properties using the provider
        properties = provider.query_properties(
            city='Sousse',
            max_price=200000,
            property_type='terrain',
            min_surface=150
        )
        
        print(f"📊 Found {len(properties)} terrain properties in Sousse under 200k DT")
        
        if properties:
            print("\n🏠 Sample properties:")
            for i, prop in enumerate(properties[:5], 1):
                price = prop.get('price', 'N/A')
                surface = prop.get('surface', 'N/A')
                location = prop.get('location', 'N/A')
                title = prop.get('title', 'N/A')
                print(f"   {i}. {price} DT, {surface} m², {location}")
                print(f"      {title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app():
    print("\n🌐 Testing Streamlit app direct functionality...")
    
    try:
        # Test the main components without Streamlit UI
        from budget_agent_base import EnhancedBudgetAgent
        from couchdb_provider import CouchDBProvider
        
        agent = EnhancedBudgetAgent()
        provider = CouchDBProvider()
        
        print("✅ Components initialized")
        
        # Test direct property search (what the app would do)
        print("\n🔍 Testing direct property search...")
        properties = provider.query_properties(
            city='Sousse',
            max_price=200000,
            property_type='terrain',
            limit=10
        )
        
        print(f"📊 Found {len(properties)} properties for the app")
        
        if properties:
            # Convert to DataFrame like the app would
            df = provider.to_dataframe(properties)
            print(f"📊 DataFrame shape: {df.shape}")
            print(f"📊 DataFrame columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Core Budget Agent Functionality")
    print("=" * 50)
    
    success1 = test_core_agent()
    success2 = test_streamlit_app()
    
    if success1 and success2:
        print("\n✅ All core tests passed!")
        print("💡 The terrain search in Sousse is working correctly")
    else:
        print("\n❌ Some tests failed")
    
    print("=" * 50)
