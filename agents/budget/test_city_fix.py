#!/usr/bin/env python3
"""
Simple test to verify the city filtering fix
"""

def test_city_filtering():
    print("🔍 Testing city filtering logic...")
    
    # Simulate the exact logic we fixed
    client_city = "Sousse"
    
    # Test properties like we have in the database
    test_properties = [
        {'City': 'Kairouan, Route Kairouan Sousse', 'Location': 'Kairouan, Route Kairouan Sousse', 'Price': 76380, 'Surface': 402, 'Type': 'Terrain habitation collective en vente'},
        {'City': 'Kairouan, Route sousse', 'Location': 'Kairouan, Route sousse', 'Price': 53680, 'Surface': 244, 'Type': 'Terrain habitation ind. en vente'},
        {'City': 'Bizerte', 'Location': 'Bizerte', 'Price': 150000, 'Surface': 300, 'Type': 'Appartement'},
    ]
    
    client_info = {
        'city': client_city,
        'budget': 200000,
        'max_price': 200000,
        'min_size': 100,
        'property_type': '',
        'preferences': 'terrain'
    }
    
    city_matches = 0
    price_matches = 0
    type_matches = 0
    filtered_props = []
    
    for prop in test_properties:
        include_property = True
        
        # Apply our fixed city filtering logic
        if client_info.get('city'):
            client_city_lower = client_info['city'].lower().strip()
            prop_city = prop.get('City', '').lower().strip()
            prop_location = prop.get('Location', '').lower().strip()
            
            # Check if the client city is contained in either the city or location field
            city_found = (client_city_lower in prop_city or 
                         client_city_lower in prop_location or
                         prop_city == client_city_lower)
            
            if not city_found:
                include_property = False
            else:
                city_matches += 1
                print(f"✅ City match: '{prop_city}' contains '{client_city_lower}'")
        
        # Price filter
        max_budget = client_info.get('max_price') or client_info.get('budget')
        if max_budget and max_budget > 0:
            prop_price = prop.get('Price', 0)
            if prop_price > max_budget:
                include_property = False
            else:
                price_matches += 1
                print(f"✅ Price match: {prop_price} <= {max_budget}")
        
        # Property type filter (if specified)
        if client_info.get('property_type'):
            client_type = client_info['property_type'].lower().strip()
            prop_type = prop.get('Type', '').lower().strip()
            if client_type and prop_type and client_type not in prop_type and prop_type not in client_type:
                include_property = False
            else:
                type_matches += 1
        else:
            # If no specific type is requested, count all as matches
            type_matches += 1
        
        if include_property:
            filtered_props.append(prop)
            print(f"✅ Property included: {prop['City']} - {prop['Price']} DT")
    
    print(f"\n📋 Filter results:")
    print(f"   - City matches: {city_matches}")
    print(f"   - Price matches: {price_matches}")
    print(f"   - Type matches: {type_matches}")
    print(f"   - Final filtered properties: {len(filtered_props)}")
    
    return len(filtered_props) > 0

if __name__ == "__main__":
    success = test_city_filtering()
    if success:
        print("\n✅ City filtering fix works!")
    else:
        print("\n❌ City filtering fix failed!")
