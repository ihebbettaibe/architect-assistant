from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import json

class BudgetAnalysis:
    def analyze_client_budget(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced budget analysis with embedding-based similarity search
        
        Args:
            client_info: Dictionary containing client requirements
                - city: Target city
                - budget: Client's stated budget  
                - preferences: Client requirements/preferences
                - min_size: Minimum surface area (optional)
                - max_price: Maximum budget (optional)
                - property_type: Preferred property type (optional)
        
        Returns:
            Comprehensive analysis with market insights and recommendations        """
        print(f"🏠 Analyzing budget for client in {client_info.get('city', 'Unknown')}")
        
        # 1. Find similar properties using embeddings
        similar_props = self._find_similar_properties_with_embeddings(client_info)
        
        # 2. Apply filters and get comparable properties
        comparable_props = self._filter_comparable_properties(client_info, similar_props)
        
        # 3. Calculate comprehensive market statistics
        market_stats = self._calculate_enhanced_market_stats(comparable_props, client_info)
        
        # 4. Generate AI-powered budget analysis
        budget_analysis = self._generate_enhanced_budget_analysis(
            client_info, market_stats, comparable_props        )
        
        # 5. Create budget visualization (commented out)
        # budget_viz = self._create_budget_visualization(client_info, comparable_props)
        budget_viz = None  # Disabled for now
        
        return {
            'client_info': client_info,
            'market_statistics': market_stats,
            'budget_analysis': budget_analysis,
            'comparable_properties': comparable_props[:10],  # Top 10
            'budget_visualization': budget_viz,
            'total_properties_analyzed': len(similar_props)
        }

    def _find_similar_properties_with_embeddings(self, client_info: Dict[str, Any]) -> List[Dict]:
        """Use embedding similarity or CouchDB queries to find relevant properties"""
        
        # If using CouchDB, use direct queries for efficiency
        if hasattr(self, 'use_couchdb') and self.use_couchdb and hasattr(self, 'couchdb_provider'):
            return self._find_properties_with_couchdb(client_info)
        
        # Fallback to embedding-based search
        return self._find_properties_with_embeddings_legacy(client_info)
    
    def _find_properties_with_couchdb(self, client_info: Dict[str, Any]) -> List[Dict]:
        """Use CouchDB direct queries to find relevant properties"""
        print("🔍 Using CouchDB direct queries for property search...")
        
        city = client_info.get('city')
        max_budget = client_info.get('budget') or client_info.get('max_price')
        min_surface = client_info.get('min_size', 0)
        property_type = client_info.get('property_type')
        
        # Expand budget range for better results
        search_budget = None
        if max_budget:
            search_budget = max_budget * 1.5  # Search up to 150% of budget
        
        # Query CouchDB directly
        properties = self.couchdb_provider.query_properties(
            city=city,
            max_price=search_budget,
            min_surface=min_surface,
            property_type=property_type,
            limit=200
        )
        
        print(f"📊 CouchDB query found {len(properties)} properties")
        
        # Convert to format expected by rest of the analysis
        property_metadata = []
        for prop in properties:
            metadata = {
                'City': prop.get('city', prop.get('City', '')),
                'Title': prop.get('title', prop.get('Title', '')),
                'Price': float(prop.get('price', prop.get('Price', 0))),
                'Surface': float(prop.get('surface', prop.get('Surface', 0))),
                'Location': prop.get('location', prop.get('Location', '')),
                'Type': prop.get('type', prop.get('Type', '')),
                'URL': prop.get('url', prop.get('URL', '')),
                'id': prop.get('_id', ''),
                'price_per_m2': 0
            }
            
            # Calculate price per m²
            if metadata['Price'] > 0 and metadata['Surface'] > 0:
                metadata['price_per_m2'] = metadata['Price'] / metadata['Surface']
            
            property_metadata.append(metadata)
        
        print(f"📊 Converted {len(property_metadata)} properties to metadata format")
        return property_metadata
    
    def _find_properties_with_embeddings_legacy(self, client_info: Dict[str, Any]) -> List[Dict]:
        """Use embedding similarity to find relevant properties with deduplication"""
        # Create query from client info
        query_parts = []
        if client_info.get('city'):            query_parts.append(f"{client_info['city']}")
        if client_info.get('preferences'):
            query_parts.append(client_info['preferences'])
        if client_info.get('property_type'):
            query_parts.append(client_info['property_type'])
        
        query = " ".join(query_parts) if query_parts else "property"
        print(f"🔍 Embedding search query: '{query}'")
        
        # Get similar documents
        k_value = min(100, len(self.property_metadata))
        similar_docs = self.vectorstore.similarity_search(query=query, k=k_value)
        
        print(f"📊 Found {len(similar_docs)} similar properties via embeddings")
        
        # Debug: Show sample of found properties to check for duplicates
        print("🔍 Sample of embedding search results:")
        for i, doc in enumerate(similar_docs[:3]):
            prop = doc.metadata
            print(f"   {i+1}. Price: {prop.get('Price', 0):,.0f} DT, Surface: {prop.get('Surface', 0):.0f}m², Type: {prop.get('Type', 'N/A')}")        # Deduplicate properties using exact attributes
        unique_properties = {}
        for doc in similar_docs:
            prop = doc.metadata
            # Create unique key based on exact attributes to avoid losing variety
            unique_key = (
                prop.get('Price', 0),           # Exact price
                prop.get('Surface', 0),         # Exact surface
                prop.get('URL', ''),            # Exact URL (most unique identifier)
                prop.get('Title', '')[:30]      # First 30 chars of title
            )
            if unique_key not in unique_properties:
                unique_properties[unique_key] = prop
        print(f"📊 After deduplication: {len(unique_properties)} unique properties from embeddings")
        
        # If we have a city filter, add more properties from that city
        if client_info.get('city'):
            city_properties = [p for p in self.property_metadata 
                             if p.get('City', '').lower() == client_info['city'].lower()]
            # print(f"🏙️ Direct city search found {len(city_properties)} properties in {client_info['city']}")
              # Add unique city properties
            existing_keys = set(unique_properties.keys())
            added_count = 0
            for prop in city_properties:
                prop_key = (prop.get('Price'), prop.get('Surface'), prop.get('City'), prop.get('Type'))
                if prop_key not in existing_keys:
                    unique_properties[prop_key] = prop
                    added_count += 1
            
            print(f"🔄 Added {added_count} additional unique properties from city search")
        
        deduplicated_props = list(unique_properties.values())
          # Add some randomization to ensure variety in results
        import random
        if len(deduplicated_props) > 10:
            # Keep the most relevant ones but add some randomization
            deduplicated_props = deduplicated_props[:20] + random.sample(deduplicated_props[20:], min(10, len(deduplicated_props) - 20))
        
        print(f"📊 Final deduplicated properties: {len(deduplicated_props)}")
        
        # Show price distribution to verify variety
        if deduplicated_props:
            prices = [p.get('Price', 0) for p in deduplicated_props]
            print(f"💰 Price range: {min(prices):,.0f} - {max(prices):,.0f} DT")
            unique_prices = set(prices)
            print(f"💰 Number of unique prices: {len(unique_prices)}")
        
        return deduplicated_props

    def _filter_comparable_properties(self, client_info: Dict[str, Any], properties: List[Dict]) -> List[Dict]:
        """Apply client filters to properties"""
        filtered = []
        
        print(f"🔧 Filtering {len(properties)} properties with criteria:")
        print(f"   - City: {client_info.get('city', 'Any')}")
        print(f"   - Min Size: {client_info.get('min_size', 'Any')} m²")
        print(f"   - Max Price: {client_info.get('max_price', 'Any')} DT")
        print(f"   - Property Type: {client_info.get('property_type', 'Any')}")
        
        city_matches = 0
        size_matches = 0
        price_matches = 0
        type_matches = 0
        
        for prop in properties:
            include_property = True
            
            # City filter - use contains match for more flexible city matching
            if client_info.get('city'):
                client_city = client_info['city'].lower().strip()
                prop_city = prop.get('City', '').lower().strip()
                prop_location = prop.get('Location', '').lower().strip()
                
                # Check if the client city is contained in either the city or location field
                city_found = (client_city in prop_city or 
                             client_city in prop_location or
                             prop_city == client_city)
                
                if not city_found:
                    include_property = False
                else:
                    city_matches += 1
            
            # Size filter - only apply if min_size is specified and > 0
            if client_info.get('min_size') and client_info.get('min_size') > 0:
                if prop.get('Surface', 0) < client_info['min_size']:
                    include_property = False
                else:
                    size_matches += 1
            
            # Price filter - CRITICAL: Apply max_price AND budget constraints
            max_budget = client_info.get('max_price') or client_info.get('budget')
            if max_budget and max_budget > 0:
                prop_price = prop.get('Price', 0)
                if prop_price > max_budget:
                    include_property = False
                else:
                    price_matches += 1
            
            # Property type filter - make it more flexible
            if client_info.get('property_type'):
                client_type = client_info['property_type'].lower().strip()
                prop_type = prop.get('Type', '').lower().strip()
                if client_type and prop_type and client_type not in prop_type and prop_type not in client_type:
                    include_property = False
                else:
                    type_matches += 1
            if include_property:
                filtered.append(prop)
        
        print(f"📋 Filter results:")
        print(f"   - City matches: {city_matches}")
        print(f"   - Size matches: {size_matches}")
        print(f"   - Price matches: {price_matches}")
        print(f"   - Type matches: {type_matches}")
        print(f"   - Final filtered properties: {len(filtered)}")
        
        return filtered
    
    def _calculate_enhanced_market_stats(self, properties: List[Dict], client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive market statistics"""
        if not properties:
            return {
                "inventory_count": 0,
                "market_summary": "No properties found matching criteria",
                "budget_feasibility": "Unknown - insufficient data"
            }
        
        prices = [p['Price'] for p in properties]
        surfaces = [p['Surface'] for p in properties]
        price_per_m2 = [p['Price']/p['Surface'] for p in properties]
        
        client_budget = client_info.get('budget', 0)
        
        # Calculate percentiles
        price_percentiles = np.percentile(prices, [25, 50, 75, 90])
        
        # Budget feasibility
        affordable_count = sum(1 for p in prices if p <= client_budget)
        feasibility_ratio = affordable_count / len(prices) if prices else 0
        
        return {
            "inventory_count": len(properties),
            "price_stats": {
                "min": min(prices),
                "max": max(prices),
                "mean": np.mean(prices),
                "median": np.median(prices),
                "std": np.std(prices),
                "percentiles": {
                    "25th": price_percentiles[0],
                    "50th": price_percentiles[1], 
                    "75th": price_percentiles[2],
                    "90th": price_percentiles[3]
                }
            },
            "surface_stats": {
                "min": min(surfaces),
                "max": max(surfaces), 
                "mean": np.mean(surfaces),
                "median": np.median(surfaces)
            },
            "price_per_m2_stats": {
                "min": min(price_per_m2),
                "max": max(price_per_m2),
                "mean": np.mean(price_per_m2),
                "median": np.median(price_per_m2)
            },
            "budget_feasibility": {
                "affordable_properties": affordable_count,
                "feasibility_ratio": feasibility_ratio,
                "budget_percentile": (sum(1 for p in prices if p < client_budget) / len(prices)) * 100
            }
        }
    
    def _generate_enhanced_budget_analysis(
        self, 
        client_info: Dict[str, Any], 
        market_stats: Dict[str, Any], 
        comparable_props: List[Dict]
    ) -> Dict[str, Any]:
        """Generate comprehensive AI-powered budget analysis"""
        
        if not comparable_props:
            return {
                "budget_validation": "insufficient_data",
                "recommendations": "No comparable properties found. Consider expanding search criteria.",
                "confidence_score": 0.0,
                "market_insights": "Insufficient market data for analysis."
            }
        
        # Prepare context for AI analysis
        budget = client_info.get('budget', 0)
        city = client_info.get('city', 'Unknown')
        
        market_summary = f"""
        Market Analysis for {city}:
        - Available Properties: {market_stats['inventory_count']}
        - Price Range: {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT
        - Average Price: {market_stats['price_stats']['mean']:,.0f} DT
        - Median Price: {market_stats['price_stats']['median']:,.0f} DT
        - Average Price/m²: {market_stats['price_per_m2_stats']['mean']:,.0f} DT/m²
        - Budget Feasibility: {market_stats['budget_feasibility']['feasibility_ratio']:.1%} of properties are affordable
        - Client Budget Percentile: {market_stats['budget_feasibility']['budget_percentile']:.0f}th percentile
        """
        
        # Sample properties for context
        sample_props = comparable_props[:5]
        properties_context = "\n".join([
            f"• {p.get('Title', 'Property')}: {p.get('Price', 0):,.0f} DT, {p.get('Surface', 0):.0f}m²"
            for p in sample_props
        ])
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are an expert real estate budget advisor analyzing the Tunisian property market.
                        
                        Client Profile:
                        - Target City: {city}
                        - Budget: {budget:,.0f} DT
                        - Preferences: {client_info.get('preferences', 'Not specified')}
                        - Minimum Size: {client_info.get('min_size', 'Not specified')} m²
                        - Property Type: {client_info.get('property_type', 'Not specified')}
                        
                        {market_summary}
                        
                        Sample Properties:
                        {properties_context}
                        
                        Provide analysis in JSON format with these exact keys:
                        {{
                            "budget_validation": "realistic/optimistic/conservative/insufficient",
                            "market_position": "description of where client budget sits in market",
                            "recommendations": "specific actionable advice",
                            "price_negotiation_tips": "negotiation strategies based on market data",
                            "alternative_suggestions": "alternatives if budget is challenging",
                            "market_trends": "insights about the local market",
                            "risk_assessment": "potential risks and considerations"
                        }}
                        """
                    },
                    {
                        "role": "user", 
                        "content": "Analyze my budget and provide comprehensive recommendations in JSON format."
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)            
        except Exception as e:
            # print(f"AI analysis failed: {e}")
            analysis = {
                "budget_validation": "realistic" if market_stats['budget_feasibility']['feasibility_ratio'] > 0.3 else "optimistic",
                "recommendations": "Consider expanding search criteria or negotiating on price.",
                "market_insights": "Analysis completed with basic market data."
            }
        
        # Calculate confidence score
        confidence = min(0.95, market_stats['inventory_count'] / 20)
        analysis["confidence_score"] = confidence
        
        return analysis

    def generate_market_report(self, city: str = None) -> Dict[str, Any]:
        """Generate comprehensive market report"""
        if city:
            properties = [p for p in self.property_metadata if p.get('City', '').lower() == city.lower()]
        else:
            properties = self.property_metadata
        
        if not properties:
            return {"error": "No properties found for the specified criteria"}
        
        # Calculate statistics
        prices = [p['Price'] for p in properties]
        surfaces = [p['Surface'] for p in properties]
        cities = [p['City'] for p in properties]
        
        # City-wise analysis
        city_stats = {}
        for city_name in set(cities):
            city_props = [p for p in properties if p['City'] == city_name]
            city_prices = [p['Price'] for p in city_props]
            city_stats[city_name] = {
                'count': len(city_props),
                'avg_price': np.mean(city_prices),
                'min_price': min(city_prices),
                'max_price': max(city_prices)
            }
        
        fig = None  # Disabled for now
        
        return {
            'total_properties': len(properties),
            'overall_stats': {
                'avg_price': np.mean(prices),
                'median_price': np.median(prices),
                'price_range': [min(prices), max(prices)],
                'avg_surface': np.mean(surfaces)
            },
            'city_breakdown': city_stats,
            'market_visualization': fig
        }