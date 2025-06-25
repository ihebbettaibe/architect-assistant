#!/usr/bin/env python3
"""
Test script for CouchDB connection and data retrieval
Run this to verify your CouchDB setup is working correctly
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from couchdb_provider import CouchDBProvider

def test_couchdb_connection():
    """Test CouchDB connection and basic operations"""
    print("🧪 Testing CouchDB Connection...")
    print("=" * 50)
    
    try:
        # Initialize provider
        provider = CouchDBProvider()
        print("✅ CouchDB provider initialized successfully")
        
        # Test basic query (get all properties)
        print("\n📊 Testing basic property query...")
        properties = provider.get_all_properties(limit=10)
        print(f"✅ Found {len(properties)} properties in basic query")
        
        if properties:
            print("\n🏠 Sample property:")
            sample = properties[0]
            for key, value in sample.items():
                if not key.startswith('_'):  # Skip CouchDB internal fields
                    print(f"   {key}: {value}")
        else:
            print("⚠️ No properties found in database")
            print("💡 Make sure your database contains property documents")
            return False
        
        # Test city-specific query
        print("\n🏙️ Testing city-specific query (Sfax)...")
        sfax_properties = provider.get_properties_by_city("Sfax", limit=3)
        print(f"✅ Found {len(sfax_properties)} properties in Sfax")
        
        # Test budget-based query
        print("\n💰 Testing budget-based query (max 150,000 DT)...")
        budget_properties = provider.get_properties_in_budget(150000, limit=3)
        print(f"✅ Found {len(budget_properties)} properties under 150,000 DT")
        
        # Test DataFrame conversion
        print("\n📋 Testing DataFrame conversion...")
        if properties:
            df = provider.convert_to_dataframe(properties[:10])
            print(f"✅ Created DataFrame with shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            if not df.empty:
                print(f"   Price range: {df['Price'].min():,.0f} - {df['Price'].max():,.0f} DT")
                print(f"   Surface range: {df['Surface'].min():.0f} - {df['Surface'].max():.0f} m²")
        
        # Test market statistics
        print("\n📈 Testing market statistics...")
        stats = provider.get_market_statistics()
        print(f"✅ Market statistics generated:")
        print(f"   Total properties: {stats['total_properties']:,}")
        if stats['total_properties'] > 0:
            print(f"   Average price: {stats['avg_price']:,.0f} DT")
            print(f"   Average surface: {stats['avg_surface']:.0f} m²")
            print(f"   Average price/m²: {stats['avg_price_per_m2']:,.0f} DT/m²")
        
        print("\n" + "=" * 50)
        if len(properties) > 0:
            print("🎉 All tests passed! CouchDB integration is working correctly.")
            return True
        else:
            print("⚠️ Tests completed but no data found in database.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure CouchDB is running on http://127.0.0.1:5984")
        print("   2. Verify the database 'realstate_budget' exists")
        print("   3. Check username/password (default: admin/admin)")
        print("   4. Ensure your database has property documents")
        print("   5. Run: python setup_couchdb_indexes.py")
        return False

def test_sample_query():
    """Test the sample query from your example"""
    print("\n🔍 Testing your sample query...")
    
    try:
        provider = CouchDBProvider()
        results = provider.get_properties_in_budget(max_budget=150000, city="Sfax")
        
        print(f"📊 Your query returned {len(results)} results")
        
        if results:
            print("\n🏠 Sample results:")
            for i, prop in enumerate(results[:3], 1):
                price = prop.get('price', prop.get('Price', 'N/A'))
                surface = prop.get('surface', prop.get('Surface', 'N/A'))
                city = prop.get('city', prop.get('City', 'N/A'))
                prop_type = prop.get('type', prop.get('Type', 'N/A'))
                print(f"   {i}. Price: {price} DT, Surface: {surface} m², City: {city}, Type: {prop_type}")
        else:
            print("⚠️ No results found for Sfax under 150,000 DT")
            print("💡 This might be normal if you don't have properties matching these criteria")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample query failed: {e}")
        return False

if __name__ == "__main__":
    print("🏗️ CouchDB Real Estate Database Test")
    print("=" * 50)
    
    # Run tests
    basic_test_passed = test_couchdb_connection()
    
    if basic_test_passed:
        sample_test_passed = test_sample_query()
        
        if sample_test_passed:
            print("\n✨ Everything looks good! Your CouchDB integration is ready.")
        else:
            print("\n⚠️ Basic connection works, but sample query had issues.")
    else:
        print("\n❌ Basic connection failed. Please check your CouchDB setup.")
    
    print("\n" + "=" * 50)
