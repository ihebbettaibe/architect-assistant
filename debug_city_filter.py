"""
Debug city filtering issue
"""

import sys
import os
sys.path.append('agents')

from agents.budget.minimal_budget_agent import MinimalBudgetAgent

# Create agent
agent = MinimalBudgetAgent(data_folder="cleaned_data", use_couchdb=False)

print("🔍 Debug city filtering...")

# Check the first few properties from Grand Tunis to see their structure
grand_tunis_props = []
for prop in agent.base_agent.property_metadata:
    city = prop.get('City', '')
    if 'grand tunis' in city.lower():
        grand_tunis_props.append(prop)
        if len(grand_tunis_props) >= 5:
            break

print(f"\n📊 Found {len(grand_tunis_props)} Grand Tunis properties:")
for i, prop in enumerate(grand_tunis_props):
    print(f"   {i+1}. City: '{prop.get('City', '')}', Price: {prop.get('Price', 0):,}, Surface: {prop.get('Surface', 0)}")

# Now let's test the filtering logic manually
client_city = "Grand Tunis"
print(f"\n🔍 Testing filter with client_city = '{client_city}'")

for i, prop in enumerate(grand_tunis_props):
    client_city_lower = client_city.lower().strip()
    prop_city = prop.get('City', '').lower().strip()
    prop_location = prop.get('Location', '').lower().strip()
    
    print(f"\n   Property {i+1}:")
    print(f"     Client city: '{client_city_lower}'")
    print(f"     Prop city: '{prop_city}'")
    print(f"     Prop location: '{prop_location}'")
    
    # Test original logic
    city_found_original = (client_city_lower in prop_city or 
                          client_city_lower in prop_location or
                          prop_city == client_city_lower)
    print(f"     Original logic match: {city_found_original}")
    
    # Test new mapping logic
    city_mappings = {
        'tunis': ['grand tunis', 'tunis'],
        'grand tunis': ['grand tunis', 'tunis'],
        'ariana': ['ariana', 'grand tunis'],
        'ben arous': ['ben arous', 'grand tunis']
    }
    
    possible_cities = city_mappings.get(client_city_lower, [client_city_lower])
    print(f"     Possible cities: {possible_cities}")
    
    city_found_new = False
    for possible_city in possible_cities:
        if (possible_city in prop_city or 
            possible_city in prop_location or
            prop_city == possible_city):
            city_found_new = True
            print(f"     Match found with: '{possible_city}'")
            break
    
    print(f"     New logic match: {city_found_new}")
    
print("\n✅ Debug completed")
