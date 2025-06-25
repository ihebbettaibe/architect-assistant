#!/usr/bin/env python3
"""
Debug script to check what cities and data we actually have
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def debug_database_content():
    """Debug what's actually in the database"""
    
    provider = CouchDBProvider()
    
    print("🔍 Debugging database content...")
    
    # Get all properties to see what we have - use a larger limit
    all_properties = provider.get_all_properties(limit=200)
    print(f"📊 Total properties retrieved: {len(all_properties)}")
    
    if all_properties:
        # Analyze cities
        cities = set()
        locations = set()
        price_ranges = []
        
        print("\n📋 Sample properties:")
        for i, prop in enumerate(all_properties[:10], 1):
            city = prop.get('city', 'EMPTY')
            location = prop.get('location', 'EMPTY')
            price = prop.get('price', 0)
            surface = prop.get('surface', 0)
            
            cities.add(city)
            locations.add(location)
            price_ranges.append(price)
            
            print(f"   {i}. City: '{city}' | Location: '{location}' | Price: {price} | Surface: {surface}")
        
        print(f"\n🏙️ Unique cities found: {sorted(list(cities))}")
        print(f"📍 Unique locations found: {sorted(list(locations))}")
        print(f"💰 Price range: {min(price_ranges):.0f} - {max(price_ranges):.0f} DT")
        
        # Test specific city queries
        print(f"\n🔍 Testing specific queries...")
        
        # Test each location we found
        for location in sorted(locations):
            if location and location != 'EMPTY':
                props = provider.query_properties(city=location, limit=3)
                print(f"   {location}: {len(props)} properties found")
        
        # Test Sousse specifically
        sousse_props = provider.query_properties(city="Sousse", limit=5)
        print(f"\n🏠 Sousse properties found: {len(sousse_props)}")
        
        # Test budget ranges
        budget_200k = provider.query_properties(max_price=200000, limit=5)
        print(f"💰 Properties under 200,000 DT: {len(budget_200k)}")
        
        budget_300k = provider.query_properties(max_price=300000, limit=5)
        print(f"💰 Properties under 300,000 DT: {len(budget_300k)}")
        
        # Test combined query
        sousse_budget = provider.query_properties(city="Sousse", max_price=300000, limit=5)
        print(f"🎯 Sousse under 300,000 DT: {len(sousse_budget)}")

if __name__ == "__main__":
    debug_database_content()
