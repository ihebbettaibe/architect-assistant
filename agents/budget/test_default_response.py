#!/usr/bin/env python3
"""
Test script to verify the default response format includes article, place, price, and URL
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_default_response_format():
    """Test that budget input always returns property details by default"""
    try:
        # Import after adding to path
        from budget_streamlit_app_fixed import SimpleBudgetAgent
        
        agent = SimpleBudgetAgent()
        
        test_cases = [
            "J'ai un budget de 250000 DT",
            "Budget 300000 DT pour Tunis",
            "Je dispose de 200000 dinars",
            "Mon budget est de 400000 DT à Sousse"
        ]
        
        print("🧪 Testing Default Response Format (Article, Place, Price, URL)")
        print("=" * 70)
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: '{test_input}'")
            print("-" * 50)
            
            try:
                result = agent.process_message(test_input)
                
                print(f"✅ Budget extracted: {result.get('budget_analysis', {}).get('extracted_budget')}")
                print(f"✅ Should search: {result.get('should_search')}")
                
                # Check if response includes property format
                response = result.get('agent_response', '')
                
                # Test if using Streamlit processing
                print("🤖 AGENT RESPONSE:")
                print(response[:500] + "..." if len(response) > 500 else response)
                
                # Check for property details format
                has_property_title = "**#" in response or "PROPRIÉTÉS RECOMMANDÉES" in response
                has_location = "📍" in response
                has_price = "💰" in response and "DT" in response  
                has_url = "🔗" in response and "http" in response
                
                print(f"\n📊 Format Check:")
                print(f"   Property Title: {'✅' if has_property_title else '❌'}")
                print(f"   Location: {'✅' if has_location else '❌'}")
                print(f"   Price: {'✅' if has_price else '❌'}")
                print(f"   URL: {'✅' if has_url else '❌'}")
                
                all_checks = has_property_title and has_location and has_price and has_url
                print(f"\n🎯 DEFAULT FORMAT: {'✅ CORRECT' if all_checks else '❌ MISSING ELEMENTS'}")
                
            except Exception as e:
                print(f"❌ Error processing test: {e}")
        
        print("\n" + "=" * 70)
        print("✅ Default response format test completed!")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_default_response_format()
