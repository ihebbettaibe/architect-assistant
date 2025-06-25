"""
CouchDB Data Provider for Real Estate Budget Agent
Fetches property data from CouchDB instead of CSV files
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Import configuration
try:
    from .couchdb_config import *
except ImportError:
    # Fallback values if config file is not available
    COUCHDB_URL = 'http://127.0.0.1:5984'
    DB_NAME = 'realstate_budget'
    USERNAME = 'admin'
    PASSWORD = 'admin'
    DEFAULT_QUERY_LIMIT = 100
    MIN_VALID_PRICE = 1000
    MAX_VALID_PRICE = 10000000
    MIN_VALID_SURFACE = 10
    MAX_VALID_SURFACE = 10000

class CouchDBProvider:
    """
    CouchDB data provider for real estate properties
    """
    
    def __init__(self, 
                 couchdb_url: str = None,
                 db_name: str = None,
                 username: str = None,
                 password: str = None):
        """
        Initialize CouchDB connection
        
        Args:
            couchdb_url: CouchDB server URL (defaults to config)
            db_name: Database name (defaults to config)
            username: Username for authentication (defaults to config)
            password: Password for authentication (defaults to config)
        """
        self.couchdb_url = couchdb_url or COUCHDB_URL
        self.db_name = db_name or DB_NAME
        self.auth = (username or USERNAME, password or PASSWORD)
        self.base_url = f"{self.couchdb_url}/{self.db_name}"
        
        # Test connection
        self._test_connection()
        
        # Try to create indexes for better performance
        self._ensure_indexes()
    
    def _test_connection(self):
        """Test CouchDB connection"""
        try:
            response = requests.get(self.base_url, auth=self.auth)
            if response.status_code == 200:
                print(f"✅ Connected to CouchDB: {self.db_name}")
                return True
            else:
                print(f"⚠️ CouchDB connection issue: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to CouchDB: {e}")
            return False
    
    def _ensure_indexes(self):
        """Create necessary indexes for efficient querying"""
        indexes_to_create = [
            {
                "index": {"fields": ["price"]},
                "name": "price-index"
            },
            {
                "index": {"fields": ["city"]},
                "name": "city-index"
            },
            {
                "index": {"fields": ["surface"]},
                "name": "surface-index"
            },
            {
                "index": {"fields": ["type"]},
                "name": "type-index"
            },
            {
                "index": {"fields": ["city", "price"]},
                "name": "city-price-index"
            }
        ]
        
        for index_def in indexes_to_create:
            try:
                url = f"{self.base_url}/_index"
                response = requests.post(url, json=index_def, auth=self.auth)
                if response.status_code in [200, 201]:
                    # Index created or already exists
                    pass
                elif response.status_code == 409:
                    # Index already exists, that's fine
                    pass
                else:
                    print(f"Warning: Could not create index {index_def['name']}: {response.status_code}")
            except Exception as e:
                print(f"Warning: Index creation failed for {index_def['name']}: {e}")
    
    def _test_connection(self):
        """Test CouchDB connection"""
        try:
            response = requests.get(self.base_url, auth=self.auth)
            if response.status_code == 200:
                print(f"✅ Connected to CouchDB: {self.db_name}")
                return True
            else:
                print(f"⚠️ CouchDB connection issue: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to CouchDB: {e}")
            return False
    
    def query_properties(self, 
                        city: Optional[str] = None,
                        max_price: Optional[float] = None,
                        min_price: Optional[float] = None,
                        min_surface: Optional[float] = None,
                        max_surface: Optional[float] = None,
                        property_type: Optional[str] = None,
                        limit: int = None) -> List[Dict]:
        """
        Query properties from CouchDB with filters
        
        Args:
            city: Filter by city
            max_price: Maximum price filter
            min_price: Minimum price filter
            min_surface: Minimum surface filter
            max_surface: Maximum surface filter
            property_type: Property type filter
            limit: Maximum number of results
            
        Returns:
            List of property documents
        """
        url = f"{self.base_url}/_find"
        
        # Build selector
        selector = {}
        and_conditions = []
        
        # Handle city search
        if city:
            city_condition = {
                "$or": [
                    {"city": {"$regex": f"(?i){city}"}},
                    {"location": {"$regex": f"(?i){city}"}}
                ]
            }
            and_conditions.append(city_condition)
        
        # Handle property type search
        if property_type:
            type_condition = {
                "$or": [
                    {"type": {"$regex": f"(?i){property_type}"}},
                    {"title": {"$regex": f"(?i){property_type}"}},
                    {"description": {"$regex": f"(?i){property_type}"}}
                ]
            }
            and_conditions.append(type_condition)
        
        # Handle price filters
        if max_price is not None or min_price is not None:
            price_condition = {}
            if max_price is not None:
                price_condition["$lte"] = max_price
            if min_price is not None:
                price_condition["$gte"] = min_price
            and_conditions.append({"price": price_condition})
        
        # Handle surface filters
        if min_surface is not None or max_surface is not None:
            surface_condition = {}
            if min_surface is not None:
                surface_condition["$gte"] = min_surface
            if max_surface is not None:
                surface_condition["$lte"] = max_surface
            and_conditions.append({"surface": surface_condition})
        
        # Combine all conditions
        if len(and_conditions) > 1:
            selector = {"$and": and_conditions}
        elif len(and_conditions) == 1:
            selector = and_conditions[0]
        else:
            selector = {}  # No filters, get all documents
        
        query = {
            "selector": selector,
            "limit": limit or DEFAULT_QUERY_LIMIT
        }
        
        try:
            # If we have numeric filters (price/surface), use fallback approach
            # because CouchDB can't handle string-to-number conversion in queries
            has_numeric_filters = (max_price is not None or min_price is not None or 
                                 min_surface is not None or max_surface is not None)
            
            if has_numeric_filters:
                print("🔄 Using fallback query method due to numeric filters...")
                return self._fallback_query(city, max_price, min_price, min_surface, max_surface, property_type, limit or DEFAULT_QUERY_LIMIT)
            
            response = requests.post(url, json=query, auth=self.auth)
            if response.status_code == 200:
                result = response.json()
                return result.get('docs', [])
            else:
                print(f"Query error: {response.status_code} - {response.text}")
                # Fallback to _all_docs if selector query fails
                if selector:  # Only fallback if we had a complex selector
                    print("🔄 Falling back to _all_docs approach...")
                    return self._fallback_query(city, max_price, min_price, min_surface, max_surface, property_type, limit or DEFAULT_QUERY_LIMIT)
                return []
        except Exception as e:
            print(f"Error querying CouchDB: {e}")
            # Try fallback approach
            print("🔄 Trying fallback query method...")
            return self._fallback_query(city, max_price, min_price, min_surface, max_surface, property_type, limit or DEFAULT_QUERY_LIMIT)
    
    def _fallback_query(self, city=None, max_price=None, min_price=None, min_surface=None, max_surface=None, property_type=None, limit=100):
        """Fallback query method using _all_docs"""
        try:
            # Get all documents
            url = f"{self.base_url}/_all_docs"
            params = {"include_docs": "true", "limit": min(1000, limit * 10)}  # Get more docs to filter
            
            response = requests.get(url, params=params, auth=self.auth)
            if response.status_code != 200:
                print(f"Fallback query error: {response.status_code}")
                return []
            
            result = response.json()
            all_docs = [row['doc'] for row in result.get('rows', []) if not row['id'].startswith('_')]
            
            # Filter documents in Python
            filtered_docs = []
            for doc in all_docs:
                # Skip design documents
                if doc.get('_id', '').startswith('_design'):
                    continue
                
                # Apply filters
                if city:
                    doc_city = (doc.get('city') or doc.get('location', '')).lower()
                    if city.lower() not in doc_city:
                        continue
                
                if max_price is not None:
                    doc_price = self._extract_numeric(doc.get('price', 0))
                    if doc_price > max_price:
                        continue
                
                if min_price is not None:
                    doc_price = self._extract_numeric(doc.get('price', 0))
                    if doc_price < min_price:
                        continue
                
                if min_surface is not None:
                    doc_surface = self._extract_numeric(doc.get('surface', 0))
                    if doc_surface < min_surface:
                        continue
                
                if max_surface is not None:
                    doc_surface = self._extract_numeric(doc.get('surface', 0))
                    if doc_surface > max_surface:
                        continue
                
                if property_type:
                    # Check multiple fields for property type
                    type_text = (doc.get('type', '') + ' ' + 
                               doc.get('title', '') + ' ' + 
                               doc.get('description', '')).lower()
                    if property_type.lower() not in type_text:
                        continue
                
                filtered_docs.append(doc)
                
                # Limit results
                if len(filtered_docs) >= limit:
                    break
            
            print(f"📊 Fallback query found {len(filtered_docs)} properties")
            return filtered_docs
            
        except Exception as e:
            print(f"Fallback query failed: {e}")
            return []
    
    def get_all_properties(self, limit: int = 1000) -> List[Dict]:
        """
        Get all properties from the database
        
        Args:
            limit: Maximum number of properties to fetch
            
        Returns:
            List of all property documents
        """
        # Try simple approach first
        try:
            url = f"{self.base_url}/_all_docs"
            params = {"include_docs": "true", "limit": limit}
            
            response = requests.get(url, params=params, auth=self.auth)
            if response.status_code == 200:
                result = response.json()
                docs = []
                for row in result.get('rows', []):
                    doc = row.get('doc', {})
                    # Skip design documents and documents without required fields
                    if (not doc.get('_id', '').startswith('_design') and 
                        not doc.get('_id', '').startswith('_') and
                        (doc.get('price') or doc.get('Price'))):
                        docs.append(doc)
                
                print(f"📊 Retrieved {len(docs)} documents using _all_docs")
                return docs
            else:
                print(f"_all_docs error: {response.status_code}")
                return self.query_properties(limit=limit)
        except Exception as e:
            print(f"Error with _all_docs: {e}")
            return self.query_properties(limit=limit)
    
    def get_properties_by_city(self, city: str, limit: int = 100) -> List[Dict]:
        """
        Get properties filtered by city
        
        Args:
            city: City name
            limit: Maximum number of results
            
        Returns:
            List of properties in the specified city
        """
        return self.query_properties(city=city, limit=limit)
    
    def get_properties_in_budget(self, max_budget: float, city: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get properties within budget
        
        Args:
            max_budget: Maximum budget
            city: Optional city filter
            limit: Maximum number of results
            
        Returns:
            List of properties within budget
        """
        return self.query_properties(city=city, max_price=max_budget, limit=limit)
    
    def convert_to_dataframe(self, properties: List[Dict]) -> pd.DataFrame:
        """
        Convert CouchDB documents to pandas DataFrame
        
        Args:
            properties: List of property documents
            
        Returns:
            DataFrame with standardized columns
        """
        if not properties:
            return pd.DataFrame()
        
        # Extract data and standardize column names
        data = []
        for prop in properties:
            # Handle different possible field names and structures
            row = {
                'City': prop.get('city') or prop.get('location') or prop.get('City', ''),
                'Title': prop.get('title', prop.get('Title', prop.get('name', ''))),
                'Price': self._extract_numeric(prop.get('price', prop.get('Price', 0))),
                'Surface': self._extract_numeric(prop.get('surface', prop.get('Surface', 0))),
                'Location': prop.get('location', prop.get('Location', '')),
                'Type': prop.get('type', prop.get('Type', '')),
                'URL': prop.get('url', prop.get('URL', '')),
                'Description': prop.get('description', prop.get('Description', '')),
                'Bedrooms': self._extract_numeric(prop.get('bedrooms', prop.get('Bedrooms', 0))),
                'Bathrooms': self._extract_numeric(prop.get('bathrooms', prop.get('Bathrooms', 0))),
                'Year': self._extract_numeric(prop.get('year', prop.get('Year', 0))),
                'id': prop.get('_id', '')
            }
            
            # Calculate price per m² if both price and surface are available
            if row['Price'] > 0 and row['Surface'] > 0:
                row['price_per_m2'] = row['Price'] / row['Surface']
            else:
                row['price_per_m2'] = 0
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Clean and validate data
        df = self._clean_dataframe(df)
        
        return df
    
    def to_dataframe(self, properties: List[Dict]) -> pd.DataFrame:
        """
        Convert properties list to pandas DataFrame
        
        Args:
            properties: List of property documents
            
        Returns:
            DataFrame with standardized columns
        """
        if not properties:
            return pd.DataFrame()
        
        # Convert to standard format
        rows = []
        for prop in properties:
            row = {
                'ID': prop.get('_id', ''),
                'City': prop.get('city', '') or prop.get('location', ''),
                'Location': prop.get('location', '') or prop.get('city', ''),
                'Type': prop.get('type', '') or prop.get('title', ''),
                'Price': self._extract_numeric(prop.get('price', prop.get('Price', 0))),
                'Surface': self._extract_numeric(prop.get('surface', prop.get('Surface', 0))),
                'Title': prop.get('title', '') or prop.get('description', ''),
                'Description': prop.get('description', '') or prop.get('title', ''),
                'Bedrooms': self._extract_numeric(prop.get('bedrooms', prop.get('Bedrooms', 0))),
                'Bathrooms': self._extract_numeric(prop.get('bathrooms', prop.get('Bathrooms', 0))),
                'Year': self._extract_numeric(prop.get('year', prop.get('Year', 0))),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Clean and validate the DataFrame
        df = self._clean_dataframe(df)
        
        return df
    
    def _extract_numeric(self, value) -> float:
        """
        Extract numeric value from various formats
        
        Args:
            value: Value to convert
            
        Returns:
            Numeric value
        """
        if pd.isna(value) or value == '':
            return 0.0
        
        try:
            # Handle string numbers with commas or spaces
            if isinstance(value, str):
                # Remove common separators
                cleaned = value.replace(',', '').replace(' ', '').replace('DT', '').replace('TND', '')
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert to numeric where needed
        numeric_columns = ['Price', 'Surface', 'price_per_m2', 'Bedrooms', 'Bathrooms', 'Year']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid entries
        df = df[df['Price'] > 0]
        df = df[df['Surface'] > 0]
        
        # Recalculate price per m²
        df['price_per_m2'] = df['Price'] / df['Surface']
        
        # Remove extreme outliers
        price_q99 = df['Price'].quantile(0.99)
        price_q01 = df['Price'].quantile(0.01)
        surface_q99 = df['Surface'].quantile(0.99)
        surface_q01 = df['Surface'].quantile(0.01)
        
        df = df[
            (df['Price'] >= price_q01) & (df['Price'] <= price_q99) &
            (df['Surface'] >= surface_q01) & (df['Surface'] <= surface_q99)
        ]
        
        return df
    
    def get_market_statistics(self, city: Optional[str] = None) -> Dict[str, Any]:
        """
        Get market statistics for a city or overall
        
        Args:
            city: Optional city filter
            
        Returns:
            Dictionary with market statistics
        """
        properties = self.query_properties(city=city, limit=1000)
        df = self.convert_to_dataframe(properties)
        
        if df.empty:
            return {
                'total_properties': 0,
                'avg_price': 0,
                'avg_surface': 0,
                'avg_price_per_m2': 0,
                'price_range': {'min': 0, 'max': 0},
                'surface_range': {'min': 0, 'max': 0}
            }
        
        return {
            'total_properties': len(df),
            'avg_price': float(df['Price'].mean()),
            'avg_surface': float(df['Surface'].mean()),
            'avg_price_per_m2': float(df['price_per_m2'].mean()),
            'price_range': {
                'min': float(df['Price'].min()),
                'max': float(df['Price'].max())
            },
            'surface_range': {
                'min': float(df['Surface'].min()),
                'max': float(df['Surface'].max())
            },
            'city': city or 'All Cities'
        }

# Test the provider if run directly
if __name__ == "__main__":
    # Test the CouchDB provider
    provider = CouchDBProvider()
    
    # Test basic query
    print("Testing basic query...")
    properties = provider.query_properties(city="Sfax", max_price=150000, limit=5)
    print(f"Found {len(properties)} properties")
    
    # Test DataFrame conversion
    if properties:
        df = provider.convert_to_dataframe(properties)
        print(f"DataFrame shape: {df.shape}")
        print(df.head())
    
    # Test market statistics
    stats = provider.get_market_statistics(city="Sfax")
    print(f"Market stats: {stats}")
