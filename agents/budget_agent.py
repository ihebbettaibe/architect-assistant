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

# Load environment variables
load_dotenv()

class EnhancedBudgetAgent:
    """
    Agent Budget - Specialized AI agent for budget estimation and validation
    
    Objective: Estimate or validate client budget with targeted questions,
    inconsistency detection, and suggestions.
    
    Outputs: Budget range + reliability score in structured JSON format
    """
    def __init__(self, data_folder: str = "cleaned_data"):
        """
        Initialize the Enhanced Budget Agent with property data and embedding visualization
        
        Args:
            data_folder: Path to folder containing cleaned CSV files
        """
        self.agent_name = "Agent Budget"
        self.agent_role = "Budget Estimation and Validation Specialist"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.property_data = None
        self.embedding_matrix = None
        self.property_metadata = None
        
        # Load and index data
        self._load_and_index_data(data_folder)
    
    def _load_and_index_data(self, data_folder: str) -> None:
        """
        Load and preprocess property data into ChromaDB with embedding storage
        """
        dfs = []
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            print(f"Warning: Data folder '{data_folder}' not found. Creating sample data...")
            self._create_sample_data()
            return
            
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
            print("No valid CSV files found. Creating sample data...")
            self._create_sample_data()
            return

        self.property_data = pd.concat(dfs, ignore_index=True)
        self._create_vectorstore()
    
    def _create_sample_data(self):
        """Create sample property data for demonstration"""
        np.random.seed(42)
        
        cities = ['Tunis', 'Sousse', 'Sfax', 'Ariana', 'Ben Arous']
        property_types = ['Villa', 'Appartement', 'Terrain', 'Duplex', 'Studio']
        
        sample_data = []
        for i in range(100):
            city = np.random.choice(cities)
            prop_type = np.random.choice(property_types)
            
            # Generate realistic prices based on city and type
            base_price = {
                'Tunis': 400000, 'Sousse': 300000, 'Sfax': 250000, 
                'Ariana': 350000, 'Ben Arous': 320000
            }[city]
            
            type_multiplier = {
                'Studio': 0.4, 'Appartement': 0.7, 'Duplex': 1.2, 
                'Villa': 1.5, 'Terrain': 0.3
            }[prop_type]
            
            surface = np.random.randint(50, 500)
            price = int(base_price * type_multiplier * (surface / 150) * np.random.uniform(0.8, 1.3))
            
            sample_data.append({
                'City': city,
                'Title': f"{prop_type} {surface}m² - {city}",
                'Price': price,
                'Surface': surface,
                'Location': f"{city} Centre",
                'Type': prop_type,
                'source_file': 'sample_data.csv'
            })
        
        self.property_data = pd.DataFrame(sample_data)
        print(f"Created {len(sample_data)} sample properties")
        self._create_vectorstore()
    
    def _create_vectorstore(self):
        """Create vectorstore and store embeddings"""
        # Generate search text for each property
        self.property_data['search_text'] = self._generate_search_text(self.property_data)
        
        # Create documents for Chroma
        documents = []
        embeddings_list = []
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
                    'id': str(idx),
                    'price_per_m2': float(row['Price']) / float(row['Surface'])
                }
            )
            documents.append(doc)
            
            # Generate embedding
            embedding = self.embeddings.embed_query(row['search_text'])
            embeddings_list.append(embedding)
            metadata_list.append(doc.metadata)
        
        # Store embeddings and metadata
        self.embedding_matrix = np.array(embeddings_list)
        self.property_metadata = metadata_list
        
        # Create and populate Chroma vectorstore
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./enhanced_property_db"
        )
        
        # Add documents to vectorstore
        if documents:
            self.vectorstore.add_documents(documents)
        
        print(f"Indexed {len(documents)} properties with embeddings")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning with validation"""
        # Convert to numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['Price', 'Surface'])
        df = df[df['Price'] > 0]
        df = df[df['Surface'] > 0]
        
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
    
    def visualize_embeddings(self, method: str = 'pca', client_query: Optional[str] = None) -> go.Figure:
        """
        Visualize property embeddings using dimensionality reduction
        
        Args:
            method: 'pca', 'tsne', or 'both'
            client_query: Optional client query to highlight relevant properties
            
        Returns:
            Plotly figure with embedding visualization
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings available. Please load data first.")
        
        # Prepare data for visualization
        cities = [meta['City'] for meta in self.property_metadata]
        prices = [meta['Price'] for meta in self.property_metadata]
        surfaces = [meta['Surface'] for meta in self.property_metadata]
        types = [meta.get('Type', 'Unknown') for meta in self.property_metadata]
        
        if method in ['pca', 'both']:
            # PCA reduction
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(self.embedding_matrix)
            
            fig_pca = px.scatter(
                x=pca_coords[:, 0], 
                y=pca_coords[:, 1],
                color=cities,
                size=surfaces,
                hover_data={
                    'Price': prices,
                    'Surface': surfaces,
                    'Type': types
                },
                title='Property Embeddings Visualization (PCA)',
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
            )
        
        if method in ['tsne', 'both']:
            # t-SNE reduction
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embedding_matrix)-1))
            tsne_coords = tsne.fit_transform(self.embedding_matrix)
            
            fig_tsne = px.scatter(
                x=tsne_coords[:, 0], 
                y=tsne_coords[:, 1],
                color=cities,
                size=surfaces,
                hover_data={
                    'Price': prices,
                    'Surface': surfaces,
                    'Type': types
                },
                title='Property Embeddings Visualization (t-SNE)',
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
            )
        
        # Highlight client query if provided
        if client_query:
            query_embedding = np.array([self.embeddings.embed_query(client_query)])
            
            if method == 'pca':
                query_pca = pca.transform(query_embedding)
                fig_pca.add_trace(go.Scatter(
                    x=query_pca[:, 0], y=query_pca[:, 1],
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Client Query'
                ))
                return fig_pca
            elif method == 'tsne':
                # For t-SNE, we can't directly transform new points
                return fig_tsne
        
        if method == 'both':
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['PCA Visualization', 't-SNE Visualization']
            )
            
            # Add PCA plot
            for trace in fig_pca.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Add t-SNE plot  
            for trace in fig_tsne.data:
                fig.add_trace(trace, row=1, col=2)
            
            fig.update_layout(title='Property Embeddings: PCA vs t-SNE')
            return fig
        
        return fig_pca if method == 'pca' else fig_tsne
    
    def cluster_properties(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster properties based on their embeddings
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results and analysis
        """
        if self.embedding_matrix is None:
            raise ValueError("No embeddings available.")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embedding_matrix)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_properties = [self.property_metadata[j] for j, mask in enumerate(cluster_mask) if mask]
            
            if cluster_properties:
                cluster_analysis[f'Cluster {i}'] = {
                    'count': len(cluster_properties),
                    'avg_price': np.mean([p['Price'] for p in cluster_properties]),
                    'avg_surface': np.mean([p['Surface'] for p in cluster_properties]),
                    'cities': list(set([p['City'] for p in cluster_properties])),
                    'types': list(set([p.get('Type', 'Unknown') for p in cluster_properties]))
                }
        
        # Create visualization
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(self.embedding_matrix)
        
        fig = px.scatter(
            x=pca_coords[:, 0], 
            y=pca_coords[:, 1],
            color=[f'Cluster {label}' for label in cluster_labels],
            title=f'Property Clusters (K-means, k={n_clusters})',
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'visualization': fig,
            'cluster_centers': kmeans.cluster_centers_
        }
    
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
            Comprehensive analysis with market insights and recommendations
        """
        print(f"\n🏠 Analyzing budget for client in {client_info.get('city', 'Unknown')}")
        
        # 1. Find similar properties using embeddings
        similar_props = self._find_similar_properties_with_embeddings(client_info)
        
        # 2. Apply filters and get comparable properties
        comparable_props = self._filter_comparable_properties(client_info, similar_props)
        
        # 3. Calculate comprehensive market statistics
        market_stats = self._calculate_enhanced_market_stats(comparable_props, client_info)
        
        # 4. Generate AI-powered budget analysis
        budget_analysis = self._generate_enhanced_budget_analysis(
            client_info, market_stats, comparable_props
        )
        
        # 5. Create budget visualization
        budget_viz = self._create_budget_visualization(client_info, comparable_props)
        
        return {
            'client_info': client_info,
            'market_statistics': market_stats,
            'budget_analysis': budget_analysis,
            'comparable_properties': comparable_props[:10],  # Top 10
            'budget_visualization': budget_viz,
            'total_properties_analyzed': len(similar_props)
        }
    
    def _find_similar_properties_with_embeddings(self, client_info: Dict[str, Any]) -> List[Dict]:
        """Use embedding similarity to find relevant properties"""
        # Create query from client info
        query_parts = []
        if client_info.get('city'):
            query_parts.append(f"Property in {client_info['city']}")
        if client_info.get('preferences'):
            query_parts.append(client_info['preferences'])
        if client_info.get('property_type'):
            query_parts.append(client_info['property_type'])
        
        query = " ".join(query_parts)
        
        # Get similar documents
        similar_docs = self.vectorstore.similarity_search(
            query=query,
            k=min(50, len(self.property_metadata))  # Adjust based on available data
        )
        
        return [doc.metadata for doc in similar_docs]
    
    def _filter_comparable_properties(self, client_info: Dict[str, Any], properties: List[Dict]) -> List[Dict]:
        """Apply client filters to properties"""
        filtered = []
        
        for prop in properties:
            # City filter (case insensitive)
            if client_info.get('city'):
                if prop.get('City', '').lower() != client_info['city'].lower():
                    continue
            
            # Size filter
            if client_info.get('min_size'):
                if prop.get('Surface', 0) < client_info['min_size']:
                    continue
            
            # Price filter
            if client_info.get('max_price'):
                if prop.get('Price', 0) > client_info['max_price']:
                    continue
            
            # Property type filter
            if client_info.get('property_type'):
                if prop.get('Type', '').lower() != client_info['property_type'].lower():
                    continue
            
            filtered.append(prop)
        
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
            print(f"AI analysis failed: {e}")
            analysis = {
                "budget_validation": "realistic" if market_stats['budget_feasibility']['feasibility_ratio'] > 0.3 else "optimistic",
                "recommendations": "Consider expanding search criteria or negotiating on price.",
                "market_insights": "Analysis completed with basic market data."
            }
        
        # Calculate confidence score
        confidence = min(0.95, market_stats['inventory_count'] / 20)
        analysis["confidence_score"] = confidence
        
        return analysis
    
    def _create_budget_visualization(self, client_info: Dict[str, Any], properties: List[Dict]) -> go.Figure:
        """Create budget analysis visualization"""
        if not properties:
            return None
        
        prices = [p['Price'] for p in properties]
        surfaces = [p['Surface'] for p in properties]
        client_budget = client_info.get('budget', 0)
          # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Price Distribution vs Your Budget',
                'Price vs Surface Area', 
                'Properties Within Budget',
                'Price per m² Analysis'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # 1. Price histogram with budget line
        fig.add_trace(
            go.Histogram(x=prices, name='Price Distribution', opacity=0.7),
            row=1, col=1
        )
        fig.add_vline(x=client_budget, line_dash="dash", line_color="red", 
                     annotation_text=f"Your Budget: {client_budget:,.0f} DT",
                     row=1, col=1)
        
        # 2. Price vs Surface scatter
        colors = ['green' if p <= client_budget else 'red' for p in prices]
        fig.add_trace(
            go.Scatter(x=surfaces, y=prices, mode='markers',
                      marker=dict(color=colors), name='Properties'),
            row=1, col=2
        )
        fig.add_hline(y=client_budget, line_dash="dash", line_color="red",
                     row=1, col=2)
        
        # 3. Affordable vs Not Affordable pie chart
        affordable = sum(1 for p in prices if p <= client_budget)
        not_affordable = len(prices) - affordable
        fig.add_trace(
            go.Pie(labels=['Within Budget', 'Over Budget'], 
                   values=[affordable, not_affordable],
                   name="Budget Feasibility"),
            row=2, col=1
        )
        
        # 4. Price per m² box plot
        price_per_m2 = [p/s for p, s in zip(prices, surfaces)]
        fig.add_trace(
            go.Box(y=price_per_m2, name='Price/m²'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Budget Analysis for {client_info.get('city', 'Unknown')}",
            height=800,
            showlegend=False
        )
        
        return fig
    
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
        
        # Create market overview visualization
        fig = px.box(
            x=cities, y=prices,
            title="Price Distribution by City",
            labels={'x': 'City', 'y': 'Price (DT)'}
        )
        
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
    
    def process_client_input(self, client_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the Budget Agent - processes client input and returns structured output
        
        Args:
            client_input: Natural language input from client about budget/financial constraints
            conversation_context: Context from previous conversation or other agents
            
        Returns:
            Structured JSON output with budget analysis and questions
        """
        # Extract budget information from client input
        budget_info = self._extract_budget_from_text(client_input, conversation_context)
        
        # Generate targeted questions
        targeted_questions = self._generate_targeted_questions(budget_info, conversation_context)
        
        # Detect inconsistencies
        inconsistencies = self._detect_budget_inconsistencies(budget_info, conversation_context)
        
        # Provide budget suggestions
        suggestions = self._generate_budget_suggestions(budget_info, conversation_context)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(budget_info, inconsistencies)
        
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "timestamp": pd.Timestamp.now().isoformat(),
            "budget_analysis": {
                "extracted_budget": budget_info.get("budget", None),
                "budget_range": budget_info.get("budget_range", None),
                "budget_flexibility": budget_info.get("flexibility", "unknown"),
                "financing_status": budget_info.get("financing", "unknown")
            },
            "inconsistencies_detected": inconsistencies,
            "targeted_questions": targeted_questions,
            "suggestions": suggestions,
            "reliability_score": reliability_score,
            "confidence_level": self._get_confidence_level(reliability_score),
            "next_actions": self._suggest_next_actions(budget_info, reliability_score)
        }
    
    def _extract_budget_from_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract budget information from natural language text"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an expert at extracting budget and financial information from client conversations.
                        Extract the following information and return as JSON:
                        {
                            "budget": explicit budget amount in DT (null if not mentioned),
                            "budget_range": [min, max] if range mentioned,
                            "flexibility": "strict/flexible/negotiable/unknown",
                            "financing": "cash/loan/mixed/unknown",
                            "timeline": "urgent/normal/flexible/unknown",
                            "budget_confidence": "certain/approximate/unsure"
                        }
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Extract budget information from: {text}"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Budget extraction error: {e}")
            return {"budget": None, "flexibility": "unknown"}
    
    def _generate_targeted_questions(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """Generate targeted questions to refine budget understanding"""
        questions = []
        
        if not budget_info.get("budget") and not budget_info.get("budget_range"):
            questions.append("Avez-vous une idée du budget que vous souhaitez consacrer à ce projet ?")
            questions.append("Quel montant maximum seriez-vous prêt(e) à investir ?")
        
        if budget_info.get("financing") == "unknown":
            questions.append("Avez-vous déjà une solution de financement (fonds propres, crédit, mixte) ?")
        
        if budget_info.get("flexibility") == "unknown":
            questions.append("Votre budget est-il strict ou avez-vous une marge de manœuvre ?")
        
        if budget_info.get("timeline") == "unknown":
            questions.append("Quel est votre délai souhaité pour la réalisation du projet ?")
        
        # Add contextual questions based on project type
        if context and context.get("project_type"):
            project_type = context["project_type"]
            if project_type == "construction":
                questions.append("Avez-vous déjà le terrain ou est-il inclus dans le budget ?")
            elif project_type == "renovation":
                questions.append("Quel pourcentage du budget souhaitez-vous allouer aux gros œuvre vs finitions ?")
        
        return questions[:3]  # Return max 3 most relevant questions
    
    def _detect_budget_inconsistencies(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Detect potential inconsistencies in budget information"""
        inconsistencies = []
        budget = budget_info.get("budget", 0)
        
        if context:
            # Check budget vs project type
            project_type = context.get("project_type")
            surface = context.get("surface", 0)
            
            if budget and surface and project_type:
                budget_per_m2 = budget / surface
                
                # Tunisian construction cost benchmarks (rough estimates)
                cost_benchmarks = {
                    "construction": {"min": 800, "max": 2000},  # DT/m²
                    "renovation": {"min": 400, "max": 1200},
                    "extension": {"min": 600, "max": 1500}
                }
                
                if project_type in cost_benchmarks:
                    benchmark = cost_benchmarks[project_type]
                    if budget_per_m2 < benchmark["min"]:
                        inconsistencies.append({
                            "type": "budget_too_low",
                            "message": f"Le budget semble insuffisant pour {project_type} ({budget_per_m2:.0f} DT/m²). Coût typique: {benchmark['min']}-{benchmark['max']} DT/m²",
                            "severity": "high"
                        })
                    elif budget_per_m2 > benchmark["max"] * 1.5:
                        inconsistencies.append({
                            "type": "budget_very_high",
                            "message": f"Le budget semble très élevé pour {project_type} ({budget_per_m2:.0f} DT/m²). Vérifiez si cela inclut des éléments spéciaux.",
                            "severity": "medium"
                        })
        
        return inconsistencies
    
    def _generate_budget_suggestions(self, budget_info: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """Generate budget optimization suggestions"""
        suggestions = []
        budget = budget_info.get("budget", 0)
        
        if budget_info.get("flexibility") == "strict" and budget:
            suggestions.append("Avec un budget strict, je recommande de prévoir une marge de 10-15% pour les imprévus.")
        
        if budget_info.get("financing") == "loan":
            suggestions.append("Pour un financement par crédit, pensez à obtenir une pré-approbation avant de finaliser le projet.")
        
        if context and context.get("surface") and budget:
            surface = context["surface"]
            budget_per_m2 = budget / surface
            if budget_per_m2 < 1000:
                suggestions.append("Pour optimiser le budget, considérez une approche par phases ou des matériaux locaux.")
        
        suggestions.append("Je peux vous connecter avec des propriétés similaires pour valider votre estimation.")
        
        return suggestions
    
    def _calculate_reliability_score(self, budget_info: Dict[str, Any], inconsistencies: List[Dict]) -> float:
        """Calculate reliability score for the budget analysis"""
        score = 1.0
        
        # Reduce score for missing information
        if not budget_info.get("budget") and not budget_info.get("budget_range"):
            score -= 0.4
        
        if budget_info.get("budget_confidence") == "unsure":
            score -= 0.2
        elif budget_info.get("budget_confidence") == "approximate":
            score -= 0.1
        
        # Reduce score for inconsistencies
        for inconsistency in inconsistencies:
            if inconsistency["severity"] == "high":
                score -= 0.3
            elif inconsistency["severity"] == "medium":
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_confidence_level(self, reliability_score: float) -> str:
        """Convert reliability score to confidence level"""
        if reliability_score >= 0.8:
            return "high"
        elif reliability_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _suggest_next_actions(self, budget_info: Dict[str, Any], reliability_score: float) -> List[str]:
        """Suggest next actions based on budget analysis"""
        actions = []
        
        if reliability_score < 0.6:
            actions.append("collect_more_budget_info")
            actions.append("clarify_financing_options")
        
        if budget_info.get("budget"):
            actions.append("search_comparable_properties")
            actions.append("validate_budget_with_market_data")
        
        if reliability_score >= 0.7:
            actions.append("proceed_to_style_preferences")
            actions.append("assess_regulatory_constraints")
        
        return actions

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing Enhanced Budget Agent...")
    agent = EnhancedBudgetAgent()
    
    # Test the multi-agent interface
    print("\n🤖 Testing Multi-Agent Interface...")
    
    # Simulate client inputs
    test_inputs = [
        "Je souhaite construire une maison avec un budget de 350000 DT",
        "Mon budget est flexible, je peux aller jusqu'à 400000 DT",
        "J'ai besoin d'un crédit pour financer le projet"
    ]
    
    for i, client_input in enumerate(test_inputs, 1):
        print(f"\n� Client Input {i}: {client_input}")
        
        # Process with budget agent
        result = agent.process_client_input(client_input)
        
        print(f"🎯 Agent Response:")
        print(f"  - Budget Extracted: {result['budget_analysis']['extracted_budget']}")
        print(f"  - Reliability Score: {result['reliability_score']:.1%}")
        print(f"  - Confidence Level: {result['confidence_level']}")
        
        if result['inconsistencies_detected']:
            print(f"  - Inconsistencies: {len(result['inconsistencies_detected'])}")
        
        if result['targeted_questions']:
            print(f"  - Questions: {result['targeted_questions'][0]}")
        
        print(f"  - Next Actions: {result['next_actions']}")
    
    # Test traditional analysis (for backward compatibility)
    print("\n📊 Testing Traditional Analysis...")
    client_profile = {
        "city": "Sousse",
        "budget": 400000,
        "preferences": "terrain habitation construction",
        "min_size": 200,
        "max_price": 600000
    }
    
    # Traditional budget analysis
    budget_analysis = agent.analyze_client_budget(client_profile)
    
    print(f"\n=== TRADITIONAL BUDGET ANALYSIS ===")
    print(f"Client: {client_profile['city']} - Budget: {client_profile['budget']:,} DT")
    print(f"Properties Analyzed: {budget_analysis['total_properties_analyzed']}")
    print(f"Comparable Properties Found: {len(budget_analysis['comparable_properties'])}")
    
    market_stats = budget_analysis['market_statistics']
    if market_stats['inventory_count'] > 0:
        print(f"\n📈 Market Statistics:")
        print(f"- Price Range: {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT")
        print(f"- Average Price: {market_stats['price_stats']['mean']:,.0f} DT")
        print(f"- Budget Feasibility: {market_stats['budget_feasibility']['feasibility_ratio']:.1%}")
        
        analysis = budget_analysis['budget_analysis']
        print(f"\n🎯 AI Recommendations:")
        for key, value in analysis.items():
            if key != 'confidence_score':
                print(f"- {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 Confidence Score: {analysis.get('confidence_score', 0):.1%}")
        
        # Save visualizations
        print("\n📊 Generating Visualizations...")
        embedding_viz = agent.visualize_embeddings(method='pca', client_query="villa Tunis")
        embedding_viz.write_html("property_embeddings_visualization.html")
        print("✅ Embedding visualization saved to 'property_embeddings_visualization.html'")
        
        clusters = agent.cluster_properties(n_clusters=4)
        clusters['visualization'].write_html("property_clusters_visualization.html")
        print("✅ Clustering visualization saved to 'property_clusters_visualization.html'")
        
        if budget_analysis.get('budget_visualization'):
            budget_analysis['budget_visualization'].write_html("budget_analysis_visualization.html")
            print("✅ Budget analysis visualization saved to 'budget_analysis_visualization.html'")
    
    else:
        print("❌ No comparable properties found for analysis")
    
    print("\n🏁 Enhanced Budget Agent testing completed!")
    print("\n💡 The agent now supports both:")
    print("   1. Multi-agent architecture (process_client_input method)")
    print("   2. Traditional analysis (analyze_client_budget method)")