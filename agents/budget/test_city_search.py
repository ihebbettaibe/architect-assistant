#!/usr/bin/env python3
"""
Quick test to verify city search is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def test_city_search():
    """Test city search with known data"""
    provider = CouchDBProvider()
    
    print("🧪 Testing city searches with known data...")
    
    # Test Bizerte (we know this exists)
    bizerte_properties = provider.get_properties_by_city("Bizerte", limit=5)
    print(f"📊 Bizerte properties found: {len(bizerte_properties)}")
    
    if bizerte_properties:
        sample = bizerte_properties[0]
        print(f"   Sample: {sample.get('price', 'N/A')} DT, {sample.get('surface', 'N/A')} m², {sample.get('location', 'N/A')}")
    
    # Test with higher budget
    budget_properties = provider.get_properties_in_budget(300000, city="Bizerte")
    print(f"📊 Bizerte properties under 300,000 DT: {len(budget_properties)}")
    
    # Test general search
    all_properties = provider.get_all_properties(limit=10)
    print(f"📊 Total properties (sample): {len(all_properties)}")
    
    if all_properties:
        print("📋 Sample properties:")
        for i, prop in enumerate(all_properties[:3], 1):
            price = prop.get('price', 'N/A')
            surface = prop.get('surface', 'N/A') 
            location = prop.get('location', 'N/A')
            print(f"   {i}. {price} DT, {surface} m², {location}")

if __name__ == "__main__":
    test_city_search()
