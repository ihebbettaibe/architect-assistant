#!/usr/bin/env python3
"""
Test the exact same workflow the Streamlit app uses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from budget_agent_base import EnhancedBudgetAgent
from budget_analysis import BudgetAnalysis

def test_streamlit_workflow():
    print("🔍 Testing exact Streamlit workflow...")
    
    try:
        # Initialize like the Streamlit app does
        agent = EnhancedBudgetAgent(use_couchdb=True)
        analysis = BudgetAnalysis()
        
        # Set up client info exactly like the app
        client_info = {
            'city': 'Sousse',
            'budget': 200000,
            'max_price': 200000,
            'min_size': 100,
            'property_type': '',  # Empty like in the app
            'preferences': 'terrain'
        }
        
        print(f"Client info: {client_info}")
        
        # Call the same method the app calls
        print("\n🔍 Calling analyze_client_budget...")
        result = analysis.analyze_client_budget(client_info)
        
        print(f"Analysis result keys: {list(result.keys())}")
        
        if 'filtered_properties' in result:
            filtered = result['filtered_properties']
            print(f"📊 Filtered properties: {len(filtered)}")
            
            if filtered:
                print("\n🏠 Sample filtered properties:")
                for i, prop in enumerate(filtered[:3]):
                    city = prop.get('City', 'N/A')
                    price = prop.get('Price', 'N/A')
                    surface = prop.get('Surface', 'N/A')
                    prop_type = prop.get('Type', 'N/A')
                    print(f"  {i+1}. {city} - {price} DT - {surface} m² - {prop_type}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_streamlit_workflow()
