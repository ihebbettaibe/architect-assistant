#!/usr/bin/env python3
"""
Debug the numeric filtering in fallback query
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def debug_numeric_filtering():
    print("🔍 Debugging numeric filtering in fallback query...")
    
    try:
        provider = CouchDBProvider()
        
        # Get properties without filters first
        all_sousse_props = provider.query_properties(city='Sousse', limit=50)
        print(f"Total Sousse properties: {len(all_sousse_props)}")
        
        if all_sousse_props:
            print("\n💰 Testing price extraction on sample properties:")
            for i, prop in enumerate(all_sousse_props[:5]):
                raw_price = prop.get('price', 'N/A')
                extracted_price = provider._extract_numeric(raw_price)
                print(f"  {i+1}. Raw: '{raw_price}' -> Extracted: {extracted_price}")
                
                # Test if it would pass a 200k filter
                if extracted_price <= 200000:
                    print(f"      ✅ Would pass 200k filter")
                else:
                    print(f"      ❌ Would fail 200k filter")
        
        # Test the fallback query directly 
        print(f"\n🔄 Testing fallback query with 200k budget...")
        fallback_props = provider._fallback_query(
            city='Sousse',
            max_price=200000,
            property_type='terrain',
            limit=50
        )
        print(f"Fallback query result: {len(fallback_props)} properties")
        
        if fallback_props:
            print("✅ Fallback found properties!")
            for i, prop in enumerate(fallback_props[:3]):
                price = prop.get('price', 'N/A')
                location = prop.get('location', 'N/A')
                title = prop.get('title', 'N/A')
                print(f"  {i+1}. {price} - {location} - {title}")
        else:
            print("❌ Fallback found no properties")
            
            # Debug the fallback filtering step by step
            print("\n🔍 Debug fallback filtering step by step...")
            all_props = provider.get_all_properties(limit=100)
            print(f"Total properties loaded: {len(all_props)}")
            
            city_filtered = []
            for doc in all_props:
                if doc.get('_id', '').startswith('_design'):
                    continue
                    
                doc_city = (doc.get('city') or doc.get('location', '')).lower()
                if 'sousse' in doc_city:
                    city_filtered.append(doc)
            
            print(f"After city filter (Sousse): {len(city_filtered)}")
            
            price_filtered = []
            for doc in city_filtered:
                doc_price = provider._extract_numeric(doc.get('price', 0))
                if doc_price <= 200000 and doc_price > 0:
                    price_filtered.append(doc)
            
            print(f"After price filter (<=200k): {len(price_filtered)}")
            
            type_filtered = []
            for doc in price_filtered:
                type_text = (doc.get('type', '') + ' ' + 
                           doc.get('title', '') + ' ' + 
                           doc.get('description', '')).lower()
                if 'terrain' in type_text:
                    type_filtered.append(doc)
            
            print(f"After type filter (terrain): {len(type_filtered)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_numeric_filtering()
