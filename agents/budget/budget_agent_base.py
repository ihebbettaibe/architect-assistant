import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import Dict, Any, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Import CouchDB provider
from .couchdb_provider import CouchDBProvider
from .budget_analysis import BudgetAnalysis

# Load environment variables
load_dotenv()

class EnhancedBudgetAgent(BudgetAnalysis):
    """
    Agent Budget - Specialized AI agent for budget estimation and validation
    
    Objective: Estimate or validate client budget with targeted questions,
    inconsistency detection, and suggestions.
    
    Outputs: Budget range + reliability score in structured JSON format
    """
    def __init__(self, data_folder: str = None, use_couchdb: bool = True):
        """
        Initialize the Enhanced Budget Agent with property data and embedding visualization
        
        Args:
            data_folder: Path to folder containing cleaned CSV files (legacy support)
            use_couchdb: Whether to use CouchDB as data source (default: True)
        """
        self.agent_name = "Agent Budget"
        self.agent_role = "Budget Estimation and Validation Specialist"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.property_data = None
        self.embedding_matrix = None
        self.property_metadata = None
        self.use_couchdb = use_couchdb
        self.couchdb_provider = None
        
        # Load and index data
        if use_couchdb:
            self._load_and_index_data_from_couchdb()
        else:
            self._load_and_index_data(data_folder or "cleaned_data")
    
    def _load_and_index_data_from_couchdb(self) -> None:
        """
        Load and preprocess property data from CouchDB with simplified indexing
        """
        try:
            # Initialize CouchDB provider
            self.couchdb_provider = CouchDBProvider()
            
            # Get all properties from CouchDB
            print("📊 Loading properties from CouchDB...")
            properties = self.couchdb_provider.get_all_properties(limit=5000)
            
            if not properties:
                raise ValueError("No properties found in CouchDB database.")
            
            # Convert to DataFrame
            self.property_data = self.couchdb_provider.convert_to_dataframe(properties)
            
            if self.property_data.empty:
                raise ValueError("No valid property data after conversion and cleaning.")
            
            print(f"✅ Loaded {len(self.property_data)} properties from CouchDB")
            
            # Try to create vectorstore, but continue if it fails
            try:
                self._create_vectorstore()
            except Exception as vector_error:
                print(f"⚠️ Vectorstore creation failed: {vector_error}")
                print("📊 Continuing with basic property data (search will be limited)")
                # Set up basic metadata for fallback
                self._setup_basic_metadata()
            
        except Exception as e:
            print(f"❌ Error loading data from CouchDB: {e}")
            print("🔄 Falling back to CSV file loading...")
            # Fallback to CSV loading with proper path detection
            self.use_couchdb = False
            self._load_and_index_data_with_path_detection()
    
    def _load_and_index_data(self, data_folder: str) -> None:
        """
        Load and preprocess property data into ChromaDB with embedding storage
        """
        dfs = []
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder '{data_folder}' not found. Please ensure the cleaned_data folder exists with property CSV files.")
            
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                filepath = os.path.join(data_folder, file)
                try:
                    df = pd.read_csv(filepath)
                    
                    # Check for required columns
                    if 'Price' not in df.columns or 'Surface' not in df.columns:
                        print(f"[WARN] Skipping {file}: missing 'Price' or 'Surface' column.")
                        continue

                    print(f"Loading {file}: {len(df)} rows")
                    # Clean and standardize data
                    df = self._clean_data(df)
                    df['source_file'] = file
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        if not dfs:
            raise ValueError("No valid CSV files found with required 'Price' and 'Surface' columns.")

        self.property_data = pd.concat(dfs, ignore_index=True)
        self._create_vectorstore()
    
    def _create_vectorstore(self):
        """Create vectorstore and store embeddings with error handling"""
        try:
            # Generate search text for each property
            self.property_data['search_text'] = self._generate_search_text(self.property_data)
            
            # Create documents for Chroma
            documents = []
            metadata_list = []
            
            for idx, row in self.property_data.iterrows():
                # Create document
                doc = Document(
                    page_content=row['search_text'],
                    metadata={
                        'City': row.get('City', ''),
                        'Title': row.get('Title', ''),
                        'Price': float(row['Price']),
                        'Surface': float(row['Surface']),
                        'Location': row.get('Location', ''),
                        'Type': row.get('Type', ''),
                        'URL': row.get('URL', ''),
                        'id': str(idx),
                        'price_per_m2': float(row['Price']) / float(row['Surface'])
                    }
                )
                documents.append(doc)
                metadata_list.append(doc.metadata)
            
            # Store metadata
            self.property_metadata = metadata_list
            
            # Try to create Chroma vectorstore with error handling
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory="./enhanced_property_db"
                )
                
                # Add documents to vectorstore
                if documents:
                    self.vectorstore.add_documents(documents)
                
                print(f"Indexed {len(documents)} properties with embeddings")
                
            except Exception as chroma_error:
                print(f"⚠️ ChromaDB vectorstore creation failed: {chroma_error}")
                print("📊 Using basic embedding storage instead")
                
                # Generate embeddings manually for fallback
                embeddings_list = []
                for doc in documents:
                    embedding = self.embeddings.embed_query(doc.page_content)
                    embeddings_list.append(embedding)
                
                self.embedding_matrix = np.array(embeddings_list)
                print(f"📊 Generated {len(embeddings_list)} embeddings for fallback search")
                
        except Exception as e:
            print(f"❌ Error in vectorstore creation: {e}")
            # Set up basic metadata as fallback
            self._setup_basic_metadata()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning with validation"""
        # Convert to numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
        
        # Remove invalid surface data (keep properties even if Price is missing)
        df = df.dropna(subset=['Surface'])
        df = df[df['Surface'] > 0]
        
        # For missing prices, estimate based on average price per m² in the region
        missing_price_mask = df['Price'].isna() | (df['Price'] <= 0)
        if missing_price_mask.sum() > 0:
            print(f"⚠️ Found {missing_price_mask.sum()} properties with missing prices, estimating...")
            
            # Get valid properties for price estimation
            valid_prices = df[~missing_price_mask]
            
            if len(valid_prices) > 0:
                # Calculate average price per m² from valid data
                avg_price_per_m2 = (valid_prices['Price'] / valid_prices['Surface']).mean()
                
                # If no valid price data, use reasonable defaults by city
                if pd.isna(avg_price_per_m2):
                    city_defaults = {
                        'Grand Tunis': 2000,  # 2000 TND/m²
                        'Tunis': 2000,
                        'Sousse': 1500,
                        'Sfax': 1500,
                        'Monastir': 1400,
                        'Mahdia': 1200,
                        'Kairouan': 1000,
                        'Bizerte': 1300
                    }
                    
                    # Apply city-specific defaults
                    for city, default_price in city_defaults.items():
                        city_mask = missing_price_mask & (df['City'].str.contains(city, case=False, na=False))
                        df.loc[city_mask, 'Price'] = df.loc[city_mask, 'Surface'] * default_price
                else:
                    # Use calculated average for missing prices
                    df.loc[missing_price_mask, 'Price'] = df.loc[missing_price_mask, 'Surface'] * avg_price_per_m2
            
            print(f"✅ Estimated prices for {missing_price_mask.sum()} properties")
        
        # Now remove any remaining invalid data
        df = df.dropna(subset=['Price'])
        df = df[df['Price'] > 0]
        
        # Add calculated fields
        df['price_per_m2'] = df['Price'] / df['Surface']
        
        return df
    
    def _generate_search_text(self, df: pd.DataFrame) -> List[str]:
        """Create rich searchable text for each property"""
        texts = []
        for _, row in df.iterrows():
            text = (
                f"Property: {row.get('Type', 'Property')} in {row.get('City', 'Unknown')} "
                f"Title: {row.get('Title', 'No title')} "
                f"Price: {row['Price']:,.0f} DT "
                f"Size: {row['Surface']:.0f} m² "
                f"Price per m²: {row.get('price_per_m2', 0):,.0f} DT/m² "
                f"Location: {row.get('Location', 'N/A')} "
                f"Features: affordable housing real estate property investment"
            )
            texts.append(text)
        return texts
    
    def get_most_compatible_property(self, client_info: Dict[str, Any], comparable_properties: List[Dict]) -> Dict[str, Any]:
        """
        Find the most compatible property from the comparable properties list
        and return it with URL information
        
        Args:
            client_info: Client requirements
            comparable_properties: List of comparable properties from analysis
            
        Returns:
            Dictionary with the most compatible property including URL
        """
        if not comparable_properties:
            return None
        
        # Score each property based on client preferences
        scored_properties = []
        
        for prop in comparable_properties:
            score = 0
            max_score = 0
            
            # Budget compatibility (highest weight)
            if client_info.get('budget'):
                price_diff = abs(prop.get('Price', 0) - client_info['budget']) / client_info['budget']
                score += max(0, 1 - price_diff) * 40  # 40% weight
                max_score += 40
            
            # Size compatibility
            if client_info.get('min_size'):
                if prop.get('Surface', 0) >= client_info['min_size']:
                    size_ratio = min(1, prop.get('Surface', 0) / client_info['min_size'])
                    score += size_ratio * 30  # 30% weight
                max_score += 30
            
            # Property type match
            if client_info.get('property_type'):
                if prop.get('Type', '').lower() in client_info['property_type'].lower():
                    score += 20  # 20% weight
                max_score += 20
            
            # Price per m² efficiency
            if prop.get('price_per_m2'):
                # Lower price per m² gets higher score
                avg_price_per_m2 = np.mean([p.get('price_per_m2', 0) for p in comparable_properties])
                if avg_price_per_m2 > 0:
                    efficiency_score = max(0, 1 - (prop.get('price_per_m2', 0) / avg_price_per_m2 - 1))
                    score += efficiency_score * 10  # 10% weight
                max_score += 10
            
            # Normalize score
            final_score = score / max_score if max_score > 0 else 0
            
            scored_properties.append({
                'property': prop,
                'compatibility_score': final_score
            })
        
        # Sort by compatibility score
        scored_properties.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        # Return the most compatible property with additional info
        best_match = scored_properties[0]['property']
        
        return {
            'property_details': {
                'Title': best_match.get('Title', 'N/A'),
                'City': best_match.get('City', 'N/A'),
                'Type': best_match.get('Type', 'N/A'),
                'Price': best_match.get('Price', 0),
                'Surface': best_match.get('Surface', 0),
                'Location': best_match.get('Location', 'N/A'),
                'price_per_m2': best_match.get('price_per_m2', 0),
                'URL': best_match.get('URL', 'No URL available')
            },
            'compatibility_score': scored_properties[0]['compatibility_score'],
            'why_compatible': self._generate_compatibility_explanation(client_info, best_match, scored_properties[0]['compatibility_score'])
        }
    
    def _generate_compatibility_explanation(self, client_info: Dict[str, Any], property_info: Dict[str, Any], score: float) -> str:
        """Generate explanation for why this property is most compatible"""
        explanations = []
        
        # Budget compatibility
        if client_info.get('budget') and property_info.get('Price'):
            price_diff = abs(property_info['Price'] - client_info['budget'])
            price_diff_pct = (price_diff / client_info['budget']) * 100
            if price_diff_pct < 5:
                explanations.append("Price matches your budget perfectly")
            elif price_diff_pct < 15:
                explanations.append("Price is very close to your budget")
            elif property_info['Price'] < client_info['budget']:
                explanations.append("Price is below your budget, leaving room for other expenses")
        
        # Size compatibility
        if client_info.get('min_size') and property_info.get('Surface'):
            if property_info['Surface'] >= client_info['min_size']:
                size_excess = property_info['Surface'] - client_info['min_size']
                if size_excess > 50:
                    explanations.append(f"Offers {size_excess:.0f}m² more than your minimum requirement")
                else:
                    explanations.append("Meets your size requirements")
        
        # Type compatibility
        if client_info.get('property_type') and property_info.get('Type'):
            if property_info['Type'].lower() in client_info['property_type'].lower():
                explanations.append(f"Matches your preferred property type ({property_info['Type']})")
        
        # Location
        if client_info.get('city') and property_info.get('City'):
            if property_info['City'].lower() == client_info['city'].lower():
                explanations.append("Located in your preferred city")
        
        if not explanations:
            explanations.append("Best available option based on current market conditions")
        
        return f"Compatibility Score: {score:.1%}. " + ". ".join(explanations[:3]) + "."
    
    def _setup_basic_metadata(self):
        """Setup basic metadata when vectorstore creation fails"""
        self.property_metadata = []
        for idx, row in self.property_data.iterrows():
            metadata = {
                'City': row.get('City', ''),
                'Title': row.get('Title', ''),
                'Price': float(row['Price']),
                'Surface': float(row['Surface']),
                'Location': row.get('Location', ''),
                'Type': row.get('Type', ''),
                'URL': row.get('URL', ''),
                'id': str(idx),
                'price_per_m2': float(row['Price']) / float(row['Surface'])
            }
            self.property_metadata.append(metadata)
        print(f"📊 Set up basic metadata for {len(self.property_metadata)} properties")
    
    def _load_and_index_data_with_path_detection(self) -> None:
        """
        Load and preprocess property data with automatic path detection
        """
        # Try different possible paths for the cleaned_data folder
        current_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(current_dir, "../../cleaned_data"),  # From agents/budget/ to root/cleaned_data
            os.path.join(current_dir, "../../../cleaned_data"),  # Alternative path
            "cleaned_data",  # If running from root
            os.path.abspath(os.path.join(current_dir, "../../cleaned_data"))  # Absolute path
        ]
        
        data_folder = None
        for path in possible_paths:
            if os.path.exists(path):
                data_folder = path
                break
        
        if data_folder is None:
            print("❌ No cleaned_data folder found. Using CouchDB data only.")
            # If we have CouchDB data but no CSV files, continue with CouchDB data
            if hasattr(self, 'property_data') and not self.property_data.empty:
                print("✅ Continuing with CouchDB data (no vectorstore)")
                self._setup_basic_metadata()
                return
            else:
                raise FileNotFoundError(
                    "Data folder 'cleaned_data' not found and no CouchDB data available. "
                    "Please ensure the cleaned_data folder exists with property CSV files."
                )
        
        print(f"📁 Found data folder: {data_folder}")
        self._load_and_index_data(data_folder)