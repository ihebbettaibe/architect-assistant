#!/usr/bin/env python3
"""
Direct CouchDB query to check for Sousse data
"""

import requests

COUCHDB_URL = 'http://127.0.0.1:5984'
DB_NAME = 'realstate_budget'
USERNAME = 'admin'
PASSWORD = 'admin'

def check_sousse_data():
    """Check if Sousse data exists in CouchDB"""
    
    base_url = f"{COUCHDB_URL}/{DB_NAME}"
    auth = (USERNAME, PASSWORD)
    
    print("🔍 Checking for Sousse data in CouchDB...")
    
    # First, let's get a larger sample to see what cities we have
    url = f"{base_url}/_all_docs"
    response = requests.get(url, params={"include_docs": "true", "limit": 1000}, auth=auth)
    
    if response.status_code == 200:
        result = response.json()
        docs = result.get('rows', [])
        
        print(f"📊 Total documents: {len(docs)}")
        
        # Filter property docs and collect all locations
        property_docs = []
        locations = set()
        cities = set()
        
        for row in docs:
            if not row['id'].startswith('_'):
                doc = row['doc']
                property_docs.append(doc)
                
                location = doc.get('location', '')
                city = doc.get('city', '')
                
                if location:
                    locations.add(location)
                if city:
                    cities.add(city)
        
        print(f"🏠 Property documents: {len(property_docs)}")
        print(f"📍 All locations found: {sorted(list(locations))}")
        print(f"🏙️ All cities found: {sorted(list(cities))}")
        
        # Check specifically for Sousse in locations
        sousse_docs = [doc for doc in property_docs if 'sousse' in doc.get('location', '').lower() and not 'kairouan' in doc.get('location', '').lower()]
        print(f"\n🎯 Documents with 'Sousse' in location (excluding Kairouan routes): {len(sousse_docs)}")
        
        # Also check for exact Sousse match
        exact_sousse_docs = [doc for doc in property_docs if doc.get('location', '').lower() == 'sousse']
        print(f"🎯 Documents with exact 'Sousse' location: {len(exact_sousse_docs)}")
        
        # Check for properties that start with "Sousse"
        sousse_start_docs = [doc for doc in property_docs if doc.get('location', '').lower().startswith('sousse')]
        print(f"🎯 Documents starting with 'Sousse': {len(sousse_start_docs)}")
        
        if sousse_start_docs:
            print("📋 Properties starting with Sousse:")
            for i, doc in enumerate(sousse_start_docs[:5], 1):
                price = doc.get('price', 'N/A')
                surface = doc.get('surface', 'N/A')
                location = doc.get('location', 'N/A')
                title = doc.get('title', 'N/A')
                print(f"   {i}. {title[:50]}... | {price} DT | {surface} m² | {location}")
        
        all_sousse_docs = sousse_docs + exact_sousse_docs + sousse_start_docs
        
        # Check for properties under 200k
        budget_docs = [doc for doc in property_docs if doc.get('price', 0) <= 200000]
        print(f"\n💰 Properties under 200,000 DT: {len(budget_docs)}")
        
        # Check for Sousse under 200k
        sousse_budget_docs = [doc for doc in sousse_docs if doc.get('price', 0) <= 200000]
        print(f"🎯 Sousse properties under 200,000 DT: {len(sousse_budget_docs)}")
        
        if sousse_budget_docs:
            print("📋 Sousse properties under 200k:")
            for i, doc in enumerate(sousse_budget_docs[:3], 1):
                price = doc.get('price', 'N/A')
                surface = doc.get('surface', 'N/A')
                location = doc.get('location', 'N/A')
                print(f"   {i}. {price} DT | {surface} m² | {location}")
        
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    check_sousse_data()
