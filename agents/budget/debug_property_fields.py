#!/usr/bin/env python3
"""
Debug the property filtering issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def debug_property_fields():
    print("🔍 Debugging property field mapping...")
    
    try:
        provider = CouchDBProvider()
        
        # Get terrain properties in Sousse - first without price filter
        print("🔍 Testing query without price filter...")
        properties_no_price = provider.query_properties(
            city='Sousse',
            property_type='terrain',
            limit=10
        )
        print(f"Found {len(properties_no_price)} properties without price filter")
        
        # Get terrain properties in Sousse with price filter
        print("\n🔍 Testing query with price filter...")
        properties = provider.query_properties(
            city='Sousse',
            property_type='terrain',
            max_price=200000,
            limit=10
        )
        print(f"Found {len(properties)} properties with price filter")
        
        # Use the properties without price filter if no properties with price filter
        if not properties and properties_no_price:
            properties = properties_no_price
            print("Using properties without price filter for analysis")
        
        print(f"Found {len(properties)} properties")
        
        if properties:
            print("\n📋 First property raw data:")
            prop = properties[0]
            for key, value in prop.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
            
            print("\n📋 Converting to DataFrame format...")
            df = provider.to_dataframe(properties)
            
            if not df.empty:
                first_row = df.iloc[0]
                print("\n📋 DataFrame fields for first property:")
                for col in df.columns:
                    print(f"  {col}: {first_row[col]}")
                
                print(f"\n📊 City field values (first 5):")
                for i, row in df.head().iterrows():
                    print(f"  {i}: City='{row['City']}', Location='{row['Location']}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_property_fields()
