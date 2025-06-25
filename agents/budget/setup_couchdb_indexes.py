#!/usr/bin/env python3
"""
CouchDB Index Setup Script
Creates necessary indexes for the real estate database
"""

import requests
import json

COUCHDB_URL = 'http://127.0.0.1:5984'
DB_NAME = 'realstate_budget'
USERNAME = 'admin'
PASSWORD = 'admin'

def create_indexes():
    """Create all necessary indexes for efficient querying"""
    
    base_url = f"{COUCHDB_URL}/{DB_NAME}"
    auth = (USERNAME, PASSWORD)
    
    print("🔧 Setting up CouchDB indexes...")
    
    # Define indexes to create
    indexes = [
        {
            "index": {"fields": ["price"]},
            "name": "price-index",
            "type": "json"
        },
        {
            "index": {"fields": ["city"]},
            "name": "city-index", 
            "type": "json"
        },
        {
            "index": {"fields": ["surface"]},
            "name": "surface-index",
            "type": "json"
        },
        {
            "index": {"fields": ["type"]},
            "name": "type-index",
            "type": "json"
        },
        {
            "index": {"fields": ["city", "price"]},
            "name": "city-price-index",
            "type": "json"
        },
        {
            "index": {"fields": ["price", "surface"]},
            "name": "price-surface-index",
            "type": "json"
        }
    ]
    
    created_count = 0
    
    for index_def in indexes:
        try:
            url = f"{base_url}/_index"
            response = requests.post(url, json=index_def, auth=auth)
            
            if response.status_code in [200, 201]:
                result = response.json()
                if result.get('result') == 'created':
                    print(f"✅ Created index: {index_def['name']}")
                    created_count += 1
                elif result.get('result') == 'exists':
                    print(f"ℹ️ Index already exists: {index_def['name']}")
                else:
                    print(f"✅ Index ready: {index_def['name']}")
            else:
                print(f"❌ Failed to create index {index_def['name']}: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error creating index {index_def['name']}: {e}")
    
    print(f"\n📊 Summary: {created_count} new indexes created")
    
    # List all indexes
    try:
        url = f"{base_url}/_index"
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            result = response.json()
            indexes = result.get('indexes', [])
            print(f"\n📋 Total indexes in database: {len(indexes)}")
            for idx in indexes:
                print(f"   - {idx.get('name', 'unnamed')}: {idx.get('def', {}).get('fields', [])}")
        else:
            print(f"Could not list indexes: {response.status_code}")
    except Exception as e:
        print(f"Error listing indexes: {e}")

def test_sample_document():
    """Check if we have any sample documents in the database"""
    
    base_url = f"{COUCHDB_URL}/{DB_NAME}"
    auth = (USERNAME, PASSWORD)
    
    print(f"\n🔍 Checking for sample documents...")
    
    try:
        # Get all docs
        url = f"{base_url}/_all_docs"
        response = requests.get(url, params={"limit": 5, "include_docs": "true"}, auth=auth)
        
        if response.status_code == 200:
            result = response.json()
            docs = result.get('rows', [])
            
            print(f"📊 Total documents found: {len(docs)}")
            
            # Filter out design docs
            property_docs = [row for row in docs if not row['id'].startswith('_')]
            
            if property_docs:
                print(f"🏠 Property documents found: {len(property_docs)}")
                
                # Show sample document structure
                sample_doc = property_docs[0]['doc']
                print(f"\n📋 Sample document structure:")
                for key, value in sample_doc.items():
                    if not key.startswith('_'):
                        print(f"   {key}: {value}")
                        
                return True
            else:
                print("❌ No property documents found in database")
                print("💡 Make sure you have imported your real estate data into the database")
                return False
        else:
            print(f"Error checking documents: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error checking documents: {e}")
        return False

if __name__ == "__main__":
    print("🏗️ CouchDB Real Estate Database Setup")
    print("=" * 50)
    
    # Test connection first
    try:
        response = requests.get(f"{COUCHDB_URL}/{DB_NAME}", auth=(USERNAME, PASSWORD))
        if response.status_code == 200:
            print(f"✅ Connected to database: {DB_NAME}")
        else:
            print(f"❌ Cannot connect to database: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        exit(1)
    
    # Check for sample documents
    has_data = test_sample_document()
    
    if has_data:
        # Create indexes
        create_indexes()
        print("\n🎉 Setup complete!")
        print("💡 You can now run: python test_couchdb.py")
    else:
        print("\n⚠️ Database appears to be empty")
        print("💡 Please import your real estate data first")
    
    print("=" * 50)
