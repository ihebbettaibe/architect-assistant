#!/usr/bin/env python3
"""
Test CouchDB query for Sousse properties with detailed debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider
import pandas as pd

def test_sousse_search():
    print("🔍 Testing Sousse property search with detailed debugging...")
    
    try:
        # Initialize CouchDB provider
        provider = CouchDBProvider()
        print("✅ CouchDB provider initialized")
        
        # First, let's get ALL properties to see the data structure
        print("\n📊 Getting sample properties to check structure...")
        all_props = provider.get_all_properties(limit=50)
        print(f"Total properties retrieved: {len(all_props)}")
        
        if len(all_props) > 0:
            # Check first few properties to see city/location fields
            print("\n🔍 Checking first 10 properties for city/location fields:")
            for i, prop in enumerate(all_props[:10]):
                city = prop.get('city', 'N/A')
                location = prop.get('location', 'N/A')
                prop_type = prop.get('type', 'N/A')
                price = prop.get('price', 'N/A')
                print(f"  {i+1}. City: '{city}' | Location: '{location}' | Type: '{prop_type}' | Price: '{price}'")
        
        # Search for properties with "Sousse" in any field
        print("\n🔍 Searching for 'Sousse' in any field...")
        sousse_props = []
        for prop in all_props:
            prop_str = str(prop).lower()
            if 'sousse' in prop_str:
                sousse_props.append(prop)
        
        print(f"Properties containing 'sousse' anywhere: {len(sousse_props)}")
        
        if sousse_props:
            print("\n📋 Sousse properties found:")
            for i, prop in enumerate(sousse_props[:5]):  # Show first 5
                city = prop.get('city', 'N/A')
                location = prop.get('location', 'N/A')
                prop_type = prop.get('type', 'N/A')
                price = prop.get('price', 'N/A')
                print(f"  {i+1}. City: '{city}' | Location: '{location}' | Type: '{prop_type}' | Price: '{price}'")
        
        # Now test the actual CouchDB query
        print("\n🔍 Testing CouchDB query for 'Sousse'...")
        db_sousse_props = provider.query_properties(city="Sousse", limit=50)
        print(f"Properties found via CouchDB query: {len(db_sousse_props)}")
        
        if db_sousse_props:
            print("\n📋 First 5 properties from CouchDB query:")
            for i, prop in enumerate(db_sousse_props[:5]):
                city = prop.get('city', 'N/A')
                location = prop.get('location', 'N/A')
                prop_type = prop.get('type', 'N/A')
                price = prop.get('price', 'N/A')
                print(f"  {i+1}. City: '{city}' | Location: '{location}' | Type: '{prop_type}' | Price: '{price}'")
        
        # Test terrain search
        print("\n🔍 Testing terrain search in Sousse...")
        terrain_props = provider.query_properties(city="Sousse", property_type="terrain", limit=50)
        print(f"Terrain properties in Sousse: {len(terrain_props)}")
        
        if terrain_props:
            print("\n📋 Terrain properties found:")
            for i, prop in enumerate(terrain_props[:3]):
                city = prop.get('city', 'N/A')
                location = prop.get('location', 'N/A')
                prop_type = prop.get('type', 'N/A')
                price = prop.get('price', 'N/A')
                surface = prop.get('surface', 'N/A')
                print(f"  {i+1}. City: '{city}' | Location: '{location}' | Type: '{prop_type}' | Price: '{price}' | Surface: '{surface}'")
        
        # Test with budget constraint
        print("\n💰 Testing terrain in Sousse under 200k DT...")
        budget_terrain = provider.query_properties(
            city="Sousse", 
            property_type="terrain", 
            max_price=200000, 
            limit=50
        )
        print(f"Terrain in Sousse under 200k DT: {len(budget_terrain)}")
        
        if budget_terrain:
            print("\n📋 Budget terrain properties found:")
            for i, prop in enumerate(budget_terrain[:3]):
                city = prop.get('city', 'N/A')
                location = prop.get('location', 'N/A')
                prop_type = prop.get('type', 'N/A')
                price = prop.get('price', 'N/A')
                surface = prop.get('surface', 'N/A')
                print(f"  {i+1}. City: '{city}' | Location: '{location}' | Type: '{prop_type}' | Price: '{price}' | Surface: '{surface}'")
        
        # Convert to DataFrame and test
        print("\n📊 Testing DataFrame conversion...")
        try:
            if budget_terrain:
                df = provider.to_dataframe(budget_terrain)
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {list(df.columns)}")
                print(f"First few rows:")
                print(df.head())
            else:
                print("No budget terrain properties to convert to DataFrame")
        except Exception as e:
            print(f"DataFrame conversion error: {e}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sousse_search()
