#!/usr/bin/env python3
"""
Quick diagnostic script to check CouchDB document structure
"""

import requests
import json

COUCHDB_URL = 'http://127.0.0.1:5984'
DB_NAME = 'realstate_budget'
USERNAME = 'admin'
PASSWORD = 'admin'

def check_document_structure():
    """Check the actual structure of documents in the database"""
    
    base_url = f"{COUCHDB_URL}/{DB_NAME}"
    auth = (USERNAME, PASSWORD)
    
    print("🔍 Checking document structure in CouchDB...")
    
    try:
        # Get a few sample documents
        url = f"{base_url}/_all_docs"
        response = requests.get(url, params={"limit": 10, "include_docs": "true"}, auth=auth)
        
        if response.status_code == 200:
            result = response.json()
            docs = result.get('rows', [])
            
            print(f"📊 Total documents in database: {len(docs)}")
            
            # Show all documents (including design docs) to see what's there
            for i, row in enumerate(docs, 1):
                doc_id = row.get('id', 'No ID')
                print(f"   {i}. Document ID: {doc_id}")
            
            # Filter out design docs
            property_docs = [row for row in docs if not row['id'].startswith('_design')]
            
            print(f"\n📋 Non-design documents: {len(property_docs)}")
            
            if property_docs:
                print(f"📊 Found {len(property_docs)} sample documents")
                
                for i, row in enumerate(property_docs[:3], 1):  # Show first 3
                    doc = row['doc']
                    print(f"\n📋 Document {i} structure:")
                    print(f"   ID: {doc.get('_id', 'No ID')}")
                    
                    # Show all fields
                    for key, value in doc.items():
                        if not key.startswith('_'):
                            print(f"   {key}: {value}")
                    
                    print("-" * 40)
                
                # Try to identify field patterns
                sample_doc = property_docs[0]['doc']
                print(f"\n🔍 Field analysis for mapping:")
                
                # Check for common field variations
                price_fields = [k for k in sample_doc.keys() if 'price' in k.lower() or 'prix' in k.lower()]
                surface_fields = [k for k in sample_doc.keys() if 'surface' in k.lower() or 'area' in k.lower()]
                city_fields = [k for k in sample_doc.keys() if 'city' in k.lower() or 'ville' in k.lower()]
                type_fields = [k for k in sample_doc.keys() if 'type' in k.lower()]
                
                print(f"   Price fields found: {price_fields}")
                print(f"   Surface fields found: {surface_fields}")
                print(f"   City fields found: {city_fields}")
                print(f"   Type fields found: {type_fields}")
                
                return True
            else:
                print("❌ No property documents found")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_document_structure()
