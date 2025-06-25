#!/usr/bin/env python3
"""
Check property types and improve terrain detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def check_property_types():
    print("🔍 Checking property types in the database...")
    
    try:
        provider = CouchDBProvider()
        
        # Get properties with Sousse in location
        sousse_props = provider.query_properties(city="Sousse", limit=100)
        print(f"Found {len(sousse_props)} properties with Sousse in location")
        
        # Check all possible fields that might contain type information
        type_fields = {}
        description_samples = []
        
        for prop in sousse_props[:20]:  # Check first 20 properties
            # Collect all fields that might contain type info
            for field in ['type', 'Type', 'property_type', 'description', 'title', 'name']:
                value = prop.get(field, '')
                if value and value.strip():
                    if field not in type_fields:
                        type_fields[field] = []
                    type_fields[field].append(value)
            
            # Get description for terrain detection
            desc = prop.get('description', '') or prop.get('title', '') or ''
            if desc:
                description_samples.append(desc[:100])  # First 100 chars
        
        print("\n📊 Type fields found:")
        for field, values in type_fields.items():
            unique_values = list(set(values))[:10]  # Show max 10 unique values
            print(f"  {field}: {unique_values}")
        
        print("\n📋 Sample descriptions (first 100 chars):")
        for i, desc in enumerate(description_samples[:10]):
            print(f"  {i+1}. {desc}")
        
        # Look for terrain-related keywords in descriptions
        print("\n🔍 Checking for terrain keywords in descriptions...")
        terrain_keywords = ['terrain', 'land', 'lot', 'parcelle', 'foncier']
        terrain_found = 0
        
        for prop in sousse_props:
            desc = (prop.get('description', '') + ' ' + prop.get('title', '')).lower()
            if any(keyword in desc for keyword in terrain_keywords):
                terrain_found += 1
                print(f"  Terrain found: {desc[:80]}...")
                if terrain_found >= 5:  # Show max 5 examples
                    break
        
        print(f"\nTotal terrain-like properties found by description: {terrain_found}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_property_types()
