#!/usr/bin/env python3
"""
Debug test for budget agent to check if the most compatible property method works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.budget.budget_main import FullBudgetAgent

def test_budget_agent():
    print("🚀 Testing Budget Agent Debug...")
    
    # Initialize agent
    agent = FullBudgetAgent(data_folder="cleaned_data")
    
    # Test client profile
    client_profile = {
        "city": "Sousse",
        "budget": 300000,
        "preferences": "terrain villa",
        "min_size": 200,
        "max_price": 350000
    }
    
    print(f"🔍 Testing with client profile: {client_profile}")
    
    # Run analysis
    analysis = agent.analyze_client_budget(client_profile)
    
    print(f"\n📊 Analysis Results:")
    print(f"- Total Properties Analyzed: {analysis.get('total_properties_analyzed', 0)}")
    print(f"- Matching Properties: {analysis['market_statistics']['inventory_count']}")
    
    # Check if comparable_properties exists
    if analysis.get('comparable_properties'):
        print(f"- Comparable Properties Count: {len(analysis['comparable_properties'])}")
        
        # Test the get_most_compatible_property method
        print(f"\n🔍 Testing get_most_compatible_property method...")
        
        try:
            most_compatible = agent.get_most_compatible_property(client_profile, analysis['comparable_properties'])
            
            if most_compatible:
                print(f"\n✅ Most Compatible Property Found:")
                prop = most_compatible['property_details']
                print(f"- Title: {prop['Title']}")
                print(f"- Price: {prop['Price']:,.0f} DT")
                print(f"- Surface: {prop['Surface']:.0f} m²")
                print(f"- Location: {prop['Location']}")
                print(f"- URL: {prop['URL']}")
                print(f"- Compatibility Score: {most_compatible['compatibility_score']:.1%}")
                print(f"- Why Compatible: {most_compatible['why_compatible']}")
                
                # Check if URL is valid
                if prop['URL'] and prop['URL'] != 'No URL available':
                    print(f"✅ URL is available: {prop['URL']}")
                else:
                    print(f"❌ No URL available")
                    
            else:
                print(f"❌ No compatible property found")
                
        except Exception as e:
            print(f"❌ Error in get_most_compatible_property: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ No comparable_properties in analysis results")
        print(f"Available keys: {list(analysis.keys())}")

if __name__ == "__main__":
    test_budget_agent()
