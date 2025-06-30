#!/usr/bin/env python3
"""
Production-Ready Tecnocasa Automation System
Comprehensive automation for all cities and property types with daily scheduling,
removed articles detection, and enhanced monitoring.
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import couchdb
import re
import hashlib
import schedule
import threading
from collections import defaultdict

class ProductionTecnocasaAutomation:
    """Production-ready automation with full feature set"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the production automation system"""
        self.setup_logging()
        self.load_config(config_file)
        self.init_database()
        self.setup_directories()
        self.setup_session()
        self.init_stats()
        self.setup_property_types()
        self.setup_cities()
        
        self.logger.info("🏭 Production Tecnocasa automation initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Log filename with timestamp
        log_filename = log_dir / f"production_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Setup logging with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('production_automation')
        self.logger.info("🚀 Production automation logging initialized")
        
        # Keep reference to current log file
        self.current_log_file = log_filename
    
    def setup_directories(self):
        """Create necessary directories"""
        self.html_dir = Path('tecnocasa_html')
        self.html_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path('tecnocasa_data')
        self.data_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        self.snapshots_dir = Path('daily_snapshots')
        self.snapshots_dir.mkdir(exist_ok=True)
    
    def setup_session(self):
        """Setup HTTP session with headers"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })
    
    def load_config(self, config_file: str):
        """Load enhanced configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Enhanced default configuration
        self.config.setdefault('couchdb', {
            'url': 'http://localhost:5984',
            'username': 'admin',
            'password': 'admin',
            'database': 'realstate_budget'
        })
        
        self.config.setdefault('scraping', {
            'delay_between_requests': 2.0,
            'delay_between_cities': 5.0,
            'delay_between_property_types': 3.0,
            'timeout': 30,
            'max_retries': 3,
            'max_pages_per_city_type': 20,
            'enable_removed_detection': True,
            'snapshot_retention_days': 30
        })
        
        self.config.setdefault('email', {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': '',
            'to_emails': [],
            'send_daily_reports': True,
            'send_error_alerts': True
        })
        
        self.config.setdefault('scheduling', {
            'daily_run_time': '02:00',  # 2 AM
            'enable_scheduler': False,
            'run_on_weekends': True
        })
        
        self.base_url = "https://www.tecnocasa.tn"
    
    def init_database(self):
        """Initialize CouchDB connection with enhanced error handling"""
        try:
            couchdb_config = self.config['couchdb']
            
            # Build connection URL with auth
            url = couchdb_config['url']
            if couchdb_config.get('username') and couchdb_config.get('password'):
                url_parts = url.split('://')
                url = f"{url_parts[0]}://{couchdb_config['username']}:{couchdb_config['password']}@{url_parts[1]}"
            
            self.couch = couchdb.Server(url)
            
            # Get or create database
            db_name = couchdb_config['database']
            if db_name in self.couch:
                self.db = self.couch[db_name]
            else:
                self.db = self.couch.create(db_name)
            
            # Create necessary indexes
            self.create_database_indexes()
            
            self.logger.info(f"🗄️ Connected to CouchDB - database: {db_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to CouchDB: {e}")
            self.db = None
    
    def create_database_indexes(self):
        """Create necessary CouchDB indexes for efficient queries"""
        try:
            indexes = [
                {
                    "index": {"fields": ["extraction_date"]},
                    "name": "extraction_date_idx"
                },
                {
                    "index": {"fields": ["city", "property_type"]},
                    "name": "city_type_idx"
                },
                {
                    "index": {"fields": ["price"]},
                    "name": "price_idx"
                },
                {
                    "index": {"fields": ["active_status"]},
                    "name": "active_status_idx"
                }
            ]
            
            for index in indexes:
                try:
                    self.db.create_index(index["index"], name=index["name"])
                except Exception as e:
                    # Index might already exist
                    pass
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create database indexes: {e}")
    
    def init_stats(self):
        """Initialize comprehensive statistics tracking"""
        self.stats = {
            'cities_processed': 0,
            'property_types_processed': 0,
            'pages_fetched': 0,
            'properties_found': 0,
            'properties_saved': 0,
            'properties_updated': 0,
            'properties_removed': 0,
            'errors': 0,
            'blocked_requests': 0,
            'start_time': None,
            'end_time': None,
            'city_stats': defaultdict(lambda: defaultdict(int)),
            'error_log': []
        }
    
    def setup_property_types(self):
        """Setup all property types to scrape"""
        self.property_types = {
            'terrain': {
                'url_segment': 'terrain',
                'name': 'Terrain',
                'description': 'Land/Plots'
            },
            'appartement': {
                'url_segment': 'appartement',
                'name': 'Appartement',
                'description': 'Apartments'
            },
            'maison': {
                'url_segment': 'maison',
                'name': 'Maison',
                'description': 'Houses'
            },
            'villa': {
                'url_segment': 'villa',
                'name': 'Villa',
                'description': 'Villas'
            },
            'bureau': {
                'url_segment': 'bureau',
                'name': 'Bureau',
                'description': 'Offices'
            },
            'commerce': {
                'url_segment': 'commerce',
                'name': 'Commerce',
                'description': 'Commercial Properties'
            }
        }
    
    def setup_cities(self):
        """Setup all cities to scrape"""
        self.cities = {
            'grand-tunis': {'slug': 'nord-est-ne/grand-tunis', 'name': 'Grand Tunis'},
            'sousse': {'slug': 'centre-est-ce/sousse', 'name': 'Sousse'},
            'sfax': {'slug': 'centre-est-ce/sfax', 'name': 'Sfax'},
            'monastir': {'slug': 'centre-est-ce/monastir', 'name': 'Monastir'},
            'mahdia': {'slug': 'centre-est-ce/mahdia', 'name': 'Mahdia'},
            'kairouan': {'slug': 'centre-est-ce/kairouan', 'name': 'Kairouan'},
            'bizerte': {'slug': 'nord-est-ne/bizerte', 'name': 'Bizerte'},
            'nabeul': {'slug': 'nord-est-ne/nabeul', 'name': 'Nabeul'},
            'ariana': {'slug': 'nord-est-ne/ariana', 'name': 'Ariana'},
            'ben-arous': {'slug': 'nord-est-ne/ben-arous', 'name': 'Ben Arous'},
            'manouba': {'slug': 'nord-est-ne/manouba', 'name': 'Manouba'},
            'zaghouan': {'slug': 'centre-est-ce/zaghouan', 'name': 'Zaghouan'},
            'siliana': {'slug': 'nord-ouest-no/siliana', 'name': 'Siliana'},
            'beja': {'slug': 'nord-ouest-no/beja', 'name': 'Beja'},
            'jendouba': {'slug': 'nord-ouest-no/jendouba', 'name': 'Jendouba'},
            'kef': {'slug': 'nord-ouest-no/kef', 'name': 'Kef'},
            'kasserine': {'slug': 'centre-ouest-co/kasserine', 'name': 'Kasserine'},
            'sidi-bouzid': {'slug': 'centre-ouest-co/sidi-bouzid', 'name': 'Sidi Bouzid'},
            'gafsa': {'slug': 'sud-ouest-so/gafsa', 'name': 'Gafsa'},
            'tozeur': {'slug': 'sud-ouest-so/tozeur', 'name': 'Tozeur'},
            'kebili': {'slug': 'sud-ouest-so/kebili', 'name': 'Kebili'},
            'gabes': {'slug': 'sud-est-se/gabes', 'name': 'Gabes'},
            'medenine': {'slug': 'sud-est-se/medenine', 'name': 'Medenine'},
            'tataouine': {'slug': 'sud-est-se/tataouine', 'name': 'Tataouine'}
        }
    
    def fetch_pages_for_city_and_type(self, city_key: str, property_type_key: str, max_pages: int = None) -> List[str]:
        """Fetch HTML pages for a specific city and property type combination"""
        city_data = self.cities[city_key]
        property_type_data = self.property_types[property_type_key]
        
        city_name = city_data['name']
        property_type_name = property_type_data['name']
        city_slug = city_data['slug']
        property_type_slug = property_type_data['url_segment']
        
        if max_pages is None:
            max_pages = self.config['scraping']['max_pages_per_city_type']
        
        html_files = []
        page = 1
        
        self.logger.info(f"🏙️ Fetching {property_type_name} in {city_name} (max {max_pages} pages)")
        
        while page <= max_pages:
            url = f"{self.base_url}/vendre/{property_type_slug}/{city_slug}.html?page={page}"
            
            try:
                self.logger.debug(f"📄 Fetching {city_name} {property_type_name} - Page {page}")
                
                # Add delay between requests
                if page > 1:
                    delay = self.config['scraping']['delay_between_requests']
                    time.sleep(delay)
                
                # Fetch page with retry logic
                response = None
                for attempt in range(self.config['scraping']['max_retries']):
                    try:
                        response = self.session.get(url, timeout=self.config['scraping']['timeout'])
                        response.raise_for_status()
                        break
                    except requests.RequestException as e:
                        if attempt == self.config['scraping']['max_retries'] - 1:
                            raise
                        self.logger.warning(f"⚠️ Retry {attempt + 1}/{self.config['scraping']['max_retries']} for {url}")
                        time.sleep(2 ** attempt)
                
                # Check for blocking or CAPTCHA
                if self.is_blocked_response(response):
                    self.logger.warning(f"🚫 Detected blocking/CAPTCHA for {city_name} {property_type_name}, stopping")
                    self.stats['blocked_requests'] += 1
                    break
                
                # Parse to check if we have listings
                soup = BeautifulSoup(response.text, 'html.parser')
                property_count = self.count_properties_in_soup(soup)
                
                if property_count == 0:
                    self.logger.info(f"📭 No listings found on page {page} for {city_name} {property_type_name}, stopping")
                    break
                
                # Save HTML file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_filename = self.html_dir / f"{city_key}_{property_type_key}_page{page}_{timestamp}.html"
                
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                html_files.append(str(html_filename))
                self.stats['pages_fetched'] += 1
                self.stats['city_stats'][city_key][f'{property_type_key}_pages'] += 1
                
                self.logger.info(f"✅ Saved {property_count} {property_type_name} from {city_name} page {page}")
                page += 1
                
            except Exception as e:
                error_msg = f"Error fetching page {page} for {city_name} {property_type_name}: {e}"
                self.logger.error(f"❌ {error_msg}")
                self.stats['errors'] += 1
                self.stats['error_log'].append(error_msg)
                break
        
        return html_files
    
    def is_blocked_response(self, response: requests.Response) -> bool:
        """Enhanced blocking detection"""
        if response.status_code in [403, 429, 503]:
            return True
        
        text_lower = response.text.lower()
        blocking_indicators = [
            'captcha', 'access denied', 'blocked', 'too many requests',
            'rate limit', 'please verify', 'human verification',
            'cloudflare', 'bot detected', 'suspicious activity'
        ]
        
        return any(indicator in text_lower for indicator in blocking_indicators)
    
    def count_properties_in_soup(self, soup: BeautifulSoup) -> int:
        """Count properties in soup with enhanced detection"""
        # First check for JSON data in estates-index component
        count = 0
        
        estates_element = soup.find('estates-index')
        if estates_element:
            estates_data = estates_element.get(':estates')
            if estates_data:
                try:
                    import html
                    decoded_data = html.unescape(estates_data)
                    estates_json = json.loads(decoded_data)
                    count = len(estates_json)
                    if count > 0:
                        return count
                except Exception:
                    pass
        
        # Fallback to HTML listings
        selectors = [
            '.property-card', '.listing-item', '.immobile-item',
            '[data-property-id]', '.property-listing', '.property-container',
            '.estate-item', '.property-box'
        ]
        
        for selector in selectors:
            listings = soup.select(selector)
            if listings:
                return len(listings)
        
        return 0
    
    def parse_html_files(self, html_files: List[str]) -> List[Dict]:
        """Parse HTML files to extract property data with enhanced extraction"""
        all_properties = []
        
        for html_file in html_files:
            try:
                self.logger.debug(f"🔍 Parsing {Path(html_file).name}")
                
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                properties = self.extract_properties_from_soup(soup, html_file)
                
                if properties:
                    all_properties.extend(properties)
                    self.logger.info(f"✅ Extracted {len(properties)} properties from {Path(html_file).name}")
                
            except Exception as e:
                error_msg = f"Error parsing {html_file}: {e}"
                self.logger.error(f"❌ {error_msg}")
                self.stats['errors'] += 1
                self.stats['error_log'].append(error_msg)
        
        return all_properties
    
    def extract_properties_from_soup(self, soup: BeautifulSoup, source_file: str) -> List[Dict]:
        """Enhanced property extraction with additional metadata"""
        properties = []
        
        # Extract city and property type from filename
        filename = Path(source_file).stem
        parts = filename.split('_')
        city_key = parts[0] if len(parts) > 0 else 'unknown'
        property_type_key = parts[1] if len(parts) > 1 else 'unknown'
        
        # Look for the estates-index component with JSON data
        estates_element = soup.find('estates-index')
        if estates_element:
            estates_data = estates_element.get(':estates')
            if estates_data:
                try:
                    import html
                    decoded_data = html.unescape(estates_data)
                    estates_json = json.loads(decoded_data)
                    
                    for estate in estates_json:
                        property_data = self.extract_property_from_json(estate, city_key, property_type_key)
                        if property_data and self.validate_property_data(property_data):
                            properties.append(property_data)
                    
                    return properties
                    
                except Exception as e:
                    self.logger.error(f"❌ Error parsing JSON estate data: {e}")
        
        # Fallback to traditional HTML parsing
        self.logger.debug("🔄 Falling back to traditional HTML parsing")
        listings = self.find_listings_in_soup(soup)
        
        for listing in listings:
            try:
                property_data = self.extract_single_property(listing, soup, city_key, property_type_key)
                if property_data and self.validate_property_data(property_data):
                    properties.append(property_data)
            except Exception as e:
                self.logger.debug(f"Error extracting property: {e}")
                continue
        
        return properties
    
    def extract_property_from_json(self, estate_json: Dict, city_key: str, property_type_key: str) -> Optional[Dict]:
        """Enhanced property extraction from JSON with full metadata"""
        try:
            property_data = {
                'extraction_date': datetime.now().isoformat(),
                'source': 'tecnocasa.tn',
                'extraction_method': 'json',
                'city': city_key,
                'property_type': property_type_key,
                'city_name': self.cities.get(city_key, {}).get('name', city_key),
                'property_type_name': self.property_types.get(property_type_key, {}).get('name', property_type_key),
                'active_status': 'active'  # New property is active
            }
            
            # Extract all available fields from JSON
            field_mappings = {
                'id': 'id',
                'title': 'title',
                'subtitle': 'subtitle',
                'detail_url': 'url',
                'price': 'price_text',
                'previous_price': 'previous_price_text',
                'surface': 'surface_text',
                'discount': 'discount_text',
                'is_discounted': 'is_discounted',
                'exclusive': 'exclusive',
                'best_offer': 'best_offer',
                'country': 'country',
                'ad_type': 'ad_type',
                'rooms': 'rooms',
                'bathrooms': 'bathrooms'
            }
            
            for json_field, prop_field in field_mappings.items():
                value = estate_json.get(json_field)
                if value is not None:
                    if isinstance(value, str):
                        property_data[prop_field] = value.strip()
                    else:
                        property_data[prop_field] = value
            
            # Parse numerical values
            if property_data.get('price_text'):
                property_data['price'] = self.parse_price_from_text(property_data['price_text'])
            
            if property_data.get('previous_price_text'):
                property_data['previous_price'] = self.parse_price_from_text(property_data['previous_price_text'])
            
            if property_data.get('surface_text'):
                property_data['surface'] = self.parse_surface_from_text(property_data['surface_text'])
            
            if property_data.get('discount_text'):
                property_data['discount'] = self.parse_price_from_text(property_data['discount_text'])
            
            # Extract agency information
            agency = estate_json.get('agency', {})
            if agency:
                property_data['agency_id'] = agency.get('id', '')
                property_data['agency_name'] = agency.get('name', '')
            
            # Extract image information
            images = estate_json.get('images', [])
            if images:
                property_data['images_count'] = len(images)
                first_image = images[0] if images else {}
                image_urls = first_image.get('url', {})
                if image_urls:
                    property_data['main_image'] = image_urls.get('card') or image_urls.get('gallery_preview')
            
            # Generate property ID
            if property_data.get('url'):
                property_data['property_id'] = self.generate_property_id(property_data['url'])
            elif property_data.get('id'):
                property_data['property_id'] = f"tc_{property_data['id']}"
            else:
                # Fallback ID generation
                id_source = f"{city_key}_{property_type_key}_{property_data.get('title', '')}_{property_data.get('price_text', '')}"
                property_data['property_id'] = self.generate_property_id(id_source)
            
            # Extract location from subtitle
            if property_data.get('subtitle'):
                property_data['location'] = property_data['subtitle']
            
            return property_data
            
        except Exception as e:
            self.logger.debug(f"Error extracting property from JSON: {e}")
            return None
    
    def extract_single_property(self, listing_element, soup: BeautifulSoup, city_key: str, property_type_key: str) -> Optional[Dict]:
        """Enhanced single property extraction with full metadata"""
        property_data = {
            'extraction_date': datetime.now().isoformat(),
            'source': 'tecnocasa.tn',
            'extraction_method': 'html',
            'city': city_key,
            'property_type': property_type_key,
            'city_name': self.cities.get(city_key, {}).get('name', city_key),
            'property_type_name': self.property_types.get(property_type_key, {}).get('name', property_type_key),
            'active_status': 'active'
        }
        
        try:
            # Extract all available information using multiple selectors
            selectors_map = {
                'title': ['h3', 'h2', '.title', '.property-title', 'a'],
                'location': ['.location', '.address', '.city', '.region', '.subtitle'],
                'description': ['.description', '.summary', '.excerpt', 'p']
            }
            
            for field, selectors in selectors_map.items():
                value = self.extract_text_by_selectors(listing_element, selectors)
                if value:
                    property_data[field] = value.strip()
            
            # Extract price
            price_text = self.extract_price_from_element(listing_element)
            if price_text:
                property_data['price_text'] = price_text
                property_data['price'] = self.parse_price_from_text(price_text)
            
            # Extract surface area
            surface_text = self.extract_surface_from_element(listing_element)
            if surface_text:
                property_data['surface_text'] = surface_text
                property_data['surface'] = self.parse_surface_from_text(surface_text)
            
            # Extract property URL
            url = self.extract_property_url(listing_element)
            if url:
                property_data['url'] = url
                property_data['property_id'] = self.generate_property_id(url)
            else:
                # Generate fallback ID
                id_source = f"{city_key}_{property_type_key}_{property_data.get('title', '')}_{property_data.get('price_text', '')}"
                property_data['property_id'] = self.generate_property_id(id_source)
            
            # Limit description length
            if property_data.get('description'):
                property_data['description'] = property_data['description'][:500]
            
            return property_data
            
        except Exception as e:
            self.logger.debug(f"Error extracting single property: {e}")
            return None
    
    def extract_text_by_selectors(self, element, selectors: List[str]) -> Optional[str]:
        """Try multiple selectors to extract text"""
        for selector in selectors:
            found = element.select_one(selector)
            if found and found.get_text(strip=True):
                return found.get_text(strip=True)
        return None
    
    def extract_price_from_element(self, element) -> Optional[str]:
        """Extract price text from element"""
        price_patterns = [
            r'\d+[\s,.]?\d*\s*(?:dt|dinar|tnd|€|eur)',
            r'prix[\s:]*\d+',
            r'\d+[\s,.]?\d*\s*(?:mille|k)',
        ]
        
        text = element.get_text()
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        price_selectors = ['.price', '.cost', '.amount', '[class*="price"]', '[class*="cost"]']
        for selector in price_selectors:
            price_elem = element.select_one(selector)
            if price_elem:
                return price_elem.get_text(strip=True)
        
        return None
    
    def extract_surface_from_element(self, element) -> Optional[str]:
        """Extract surface area text from element"""
        text = element.get_text()
        
        surface_patterns = [
            r'\d+[\s,.]?\d*\s*(?:m²|m2|metre|mètre)',
            r'surface[\s:]*\d+',
            r'\d+[\s,.]?\d*\s*(?:ha|hectare)',
        ]
        
        for pattern in surface_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def extract_property_url(self, element) -> Optional[str]:
        """Extract property URL from element"""
        link = element.find('a', href=True)
        if link:
            href = link['href']
            if href.startswith('http'):
                return href
            else:
                return urljoin(self.base_url, href)
        return None
    
    def parse_price_from_text(self, price_text: str) -> Optional[int]:
        """Enhanced price parsing"""
        if not price_text:
            return None
        
        try:
            # Remove currency symbols and extract numbers
            clean_price = re.sub(r'[^\d\s]', '', price_text)
            clean_price = clean_price.replace(' ', '')
            
            if clean_price.isdigit():
                return int(clean_price)
        except ValueError:
            pass
        
        return None
    
    def parse_surface_from_text(self, surface_text: str) -> Optional[float]:
        """Enhanced surface parsing"""
        if not surface_text:
            return None
        
        try:
            match = re.search(r'(\d+(?:[.,]\d+)?)', surface_text)
            if match:
                return float(match.group(1).replace(',', '.'))
        except ValueError:
            pass
        
        return None
    
    def find_listings_in_soup(self, soup: BeautifulSoup) -> List:
        """Find property listings in the soup"""
        selectors = [
            '.property-card', '.listing-item', '.immobile-item',
            '[data-property-id]', '.property-listing', '.property-container',
            '.estate-item', '.property-box'
        ]
        
        for selector in selectors:
            listings = soup.select(selector)
            if listings:
                return listings
        
        # Fallback: look for links that seem like property URLs
        links = soup.find_all('a', href=True)
        property_links = [
            link for link in links 
            if any(pattern in link.get('href', '') for pattern in ['/detail/', '/property/', '/vendre/'])
        ]
        
        return property_links
    
    def validate_property_data(self, property_data: Dict) -> bool:
        """Enhanced validation"""
        required_fields = ['title', 'property_id']
        
        # Check required fields
        for field in required_fields:
            if not property_data.get(field):
                return False
        
        # Additional validation
        if len(property_data.get('title', '')) < 3:
            return False
        
        return True
    
    def generate_property_id(self, source_string: str) -> str:
        """Generate unique ID for property"""
        return hashlib.md5(source_string.encode()).hexdigest()[:16]
    
    def save_properties_to_database(self, properties: List[Dict]) -> Tuple[int, int]:
        """Save properties to CouchDB and return (saved_count, updated_count)"""
        if not self.db:
            self.logger.error("❌ No database connection")
            return 0, 0
        
        saved_count = 0
        updated_count = 0
        
        for prop in properties:
            try:
                prop_id = prop['property_id']
                
                # Check if property already exists
                try:
                    existing = self.db[prop_id]
                    # Update existing property
                    prop['_id'] = prop_id
                    prop['_rev'] = existing['_rev']
                    prop['updated_date'] = datetime.now().isoformat()
                    prop['created_date'] = existing.get('created_date', datetime.now().isoformat())
                    
                    # Preserve historical data
                    if 'price_history' not in prop:
                        prop['price_history'] = existing.get('price_history', [])
                    
                    # Add current price to history if different
                    current_price = prop.get('price')
                    if current_price and (not prop['price_history'] or prop['price_history'][-1]['price'] != current_price):
                        prop['price_history'].append({
                            'price': current_price,
                            'date': datetime.now().isoformat()
                        })
                    
                    self.db.save(prop)
                    updated_count += 1
                    
                except couchdb.ResourceNotFound:
                    # Create new property
                    prop['_id'] = prop_id
                    prop['created_date'] = datetime.now().isoformat()
                    prop['price_history'] = []
                    
                    # Add initial price to history
                    if prop.get('price'):
                        prop['price_history'].append({
                            'price': prop['price'],
                            'date': datetime.now().isoformat()
                        })
                    
                    self.db.save(prop)
                    saved_count += 1
                
            except Exception as e:
                error_msg = f"Error saving property {prop.get('title', 'Unknown')}: {e}"
                self.logger.error(f"❌ {error_msg}")
                self.stats['errors'] += 1
                self.stats['error_log'].append(error_msg)
        
        return saved_count, updated_count
    
    def save_daily_snapshot(self, all_properties: List[Dict]):
        """Save daily snapshot for removed article detection"""
        try:
            snapshot_date = datetime.now().strftime('%Y%m%d')
            snapshot_file = self.snapshots_dir / f"snapshot_{snapshot_date}.json"
            
            # Create snapshot data
            snapshot_data = {
                'date': datetime.now().isoformat(),
                'total_properties': len(all_properties),
                'properties': {}
            }
            
            for prop in all_properties:
                prop_id = prop['property_id']
                snapshot_data['properties'][prop_id] = {
                    'title': prop.get('title', ''),
                    'price': prop.get('price'),
                    'city': prop.get('city'),
                    'property_type': prop.get('property_type'),
                    'url': prop.get('url'),
                    'extraction_date': prop.get('extraction_date')
                }
            
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📸 Saved daily snapshot: {snapshot_file}")
            
            # Clean old snapshots
            self.cleanup_old_snapshots()
            
        except Exception as e:
            error_msg = f"Error saving daily snapshot: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.stats['error_log'].append(error_msg)
    
    def cleanup_old_snapshots(self):
        """Remove old snapshots beyond retention period"""
        try:
            retention_days = self.config['scraping']['snapshot_retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for snapshot_file in self.snapshots_dir.glob("snapshot_*.json"):
                try:
                    # Extract date from filename
                    date_str = snapshot_file.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        snapshot_file.unlink()
                        self.logger.debug(f"🗑️ Removed old snapshot: {snapshot_file}")
                        
                except Exception as e:
                    self.logger.debug(f"Error processing snapshot file {snapshot_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error cleaning old snapshots: {e}")
    
    def detect_removed_articles(self) -> List[Dict]:
        """Detect articles that were removed from the site"""
        removed_articles = []
        
        if not self.config['scraping']['enable_removed_detection']:
            return removed_articles
        
        try:
            # Get the most recent snapshot files
            snapshot_files = sorted(self.snapshots_dir.glob("snapshot_*.json"), reverse=True)
            
            if len(snapshot_files) < 2:
                self.logger.info("📸 Not enough snapshots for removed article detection")
                return removed_articles
            
            # Compare current (today) with previous snapshot
            current_snapshot_file = snapshot_files[0]
            previous_snapshot_file = snapshot_files[1]
            
            # Load snapshots
            with open(current_snapshot_file, 'r', encoding='utf-8') as f:
                current_snapshot = json.load(f)
            
            with open(previous_snapshot_file, 'r', encoding='utf-8') as f:
                previous_snapshot = json.load(f)
            
            current_properties = set(current_snapshot['properties'].keys())
            previous_properties = set(previous_snapshot['properties'].keys())
            
            # Find removed properties
            removed_property_ids = previous_properties - current_properties
            
            for prop_id in removed_property_ids:
                removed_prop = previous_snapshot['properties'][prop_id]
                removed_articles.append({
                    'property_id': prop_id,
                    'title': removed_prop['title'],
                    'price': removed_prop['price'],
                    'city': removed_prop['city'],
                    'property_type': removed_prop['property_type'],
                    'url': removed_prop['url'],
                    'last_seen': removed_prop['extraction_date'],
                    'removal_detected': datetime.now().isoformat()
                })
            
            if removed_articles:
                self.logger.info(f"🔍 Detected {len(removed_articles)} removed articles")
                
                # Mark as removed in database
                self.mark_properties_as_removed(removed_articles)
                
                # Save removed articles report
                self.save_removed_articles_report(removed_articles)
                
            else:
                self.logger.info("✅ No removed articles detected")
            
        except Exception as e:
            error_msg = f"Error detecting removed articles: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.stats['error_log'].append(error_msg)
        
        return removed_articles
    
    def mark_properties_as_removed(self, removed_articles: List[Dict]):
        """Mark properties as removed in the database"""
        if not self.db:
            return
        
        for article in removed_articles:
            try:
                prop_id = article['property_id']
                
                # Try to find and update the property in database
                try:
                    existing = self.db[prop_id]
                    existing['active_status'] = 'removed'
                    existing['removal_date'] = datetime.now().isoformat()
                    existing['updated_date'] = datetime.now().isoformat()
                    self.db.save(existing)
                    
                    self.stats['properties_removed'] += 1
                    
                except couchdb.ResourceNotFound:
                    # Property not in database, create a record of removal
                    removed_record = article.copy()
                    removed_record['_id'] = prop_id
                    removed_record['active_status'] = 'removed'
                    removed_record['removal_date'] = datetime.now().isoformat()
                    removed_record['created_date'] = datetime.now().isoformat()
                    self.db.save(removed_record)
                    
            except Exception as e:
                self.logger.error(f"❌ Error marking property as removed: {e}")
    
    def save_removed_articles_report(self, removed_articles: List[Dict]):
        """Save report of removed articles"""
        try:
            report_date = datetime.now().strftime('%Y%m%d')
            report_file = self.reports_dir / f"removed_articles_{report_date}.json"
            
            report_data = {
                'report_date': datetime.now().isoformat(),
                'total_removed': len(removed_articles),
                'removed_articles': removed_articles
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Saved removed articles report: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving removed articles report: {e}")
    
    def save_properties_to_csv(self, properties: List[Dict], suffix: str = ""):
        """Save properties to CSV file"""
        if not properties:
            return
        
        try:
            df = pd.DataFrame(properties)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = self.data_dir / f"all_properties_{suffix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            self.logger.info(f"💾 Saved {len(properties)} properties to {csv_filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving CSV: {e}")
    
    def process_city_and_property_type(self, city_key: str, property_type_key: str) -> List[Dict]:
        """Process a specific city and property type combination"""
        try:
            city_name = self.cities[city_key]['name']
            property_type_name = self.property_types[property_type_key]['name']
            
            self.logger.info(f"🏙️ Processing {property_type_name} in {city_name}")
            
            # Fetch HTML pages
            html_files = self.fetch_pages_for_city_and_type(city_key, property_type_key)
            
            if not html_files:
                self.logger.warning(f"⚠️ No HTML files fetched for {city_name} {property_type_name}")
                return []
            
            # Parse properties from HTML files
            properties = self.parse_html_files(html_files)
            
            if properties:
                self.stats['city_stats'][city_key][f'{property_type_key}_properties'] = len(properties)
                self.logger.info(f"✅ {city_name} {property_type_name}: {len(properties)} properties found")
            
            return properties
            
        except Exception as e:
            error_msg = f"Error processing {city_key} {property_type_key}: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.stats['errors'] += 1
            self.stats['error_log'].append(error_msg)
            return []
    
    def run_full_automation(self):
        """Run the complete automation for all cities and property types"""
        self.stats['start_time'] = datetime.now()
        
        self.logger.info("🏭 Starting Production Tecnocasa Automation")
        self.logger.info("=" * 80)
        self.logger.info(f"🌍 Cities: {len(self.cities)}")
        self.logger.info(f"🏠 Property Types: {len(self.property_types)}")
        self.logger.info(f"📊 Total Combinations: {len(self.cities) * len(self.property_types)}")
        self.logger.info("=" * 80)
        
        all_properties = []
        
        # Process each city
        for city_key in self.cities.keys():
            try:
                city_properties = []
                
                # Process each property type for this city
                for property_type_key in self.property_types.keys():
                    try:
                        properties = self.process_city_and_property_type(city_key, property_type_key)
                        city_properties.extend(properties)
                        
                        # Add delay between property types
                        time.sleep(self.config['scraping']['delay_between_property_types'])
                        
                    except Exception as e:
                        error_msg = f"Error processing {city_key} {property_type_key}: {e}"
                        self.logger.error(f"❌ {error_msg}")
                        self.stats['error_log'].append(error_msg)
                
                all_properties.extend(city_properties)
                self.stats['cities_processed'] += 1
                
                if city_properties:
                    self.logger.info(f"🏙️ {self.cities[city_key]['name']}: {len(city_properties)} total properties")
                
                # Add delay between cities
                time.sleep(self.config['scraping']['delay_between_cities'])
                
            except Exception as e:
                error_msg = f"Fatal error processing city {city_key}: {e}"
                self.logger.error(f"❌ {error_msg}")
                self.stats['error_log'].append(error_msg)
        
        # Process all collected properties
        if all_properties:
            self.stats['properties_found'] = len(all_properties)
            
            # Save to database
            saved_count, updated_count = self.save_properties_to_database(all_properties)
            self.stats['properties_saved'] = saved_count
            self.stats['properties_updated'] = updated_count
            
            # Save to CSV
            self.save_properties_to_csv(all_properties, "daily_extraction")
            
            # Save daily snapshot
            self.save_daily_snapshot(all_properties)
            
            # Detect removed articles
            removed_articles = self.detect_removed_articles()
            
        else:
            self.logger.warning("⚠️ No properties found during automation")
        
        self.stats['end_time'] = datetime.now()
        
        # Generate reports
        self.generate_daily_report()
        
        # Send email report if configured
        if self.config['email']['send_daily_reports']:
            self.send_email_report()
    
    def generate_daily_report(self):
        """Generate comprehensive daily report"""
        try:
            duration = self.stats['end_time'] - self.stats['start_time']
            
            report_data = {
                'report_date': datetime.now().isoformat(),
                'automation_duration': str(duration),
                'summary': {
                    'cities_processed': self.stats['cities_processed'],
                    'property_types_processed': len(self.property_types),
                    'pages_fetched': self.stats['pages_fetched'],
                    'properties_found': self.stats['properties_found'],
                    'properties_saved': self.stats['properties_saved'],
                    'properties_updated': self.stats['properties_updated'],
                    'properties_removed': self.stats['properties_removed'],
                    'errors': self.stats['errors'],
                    'blocked_requests': self.stats['blocked_requests']
                },
                'city_breakdown': dict(self.stats['city_stats']),
                'error_log': self.stats['error_log']
            }
            
            # Calculate success rate
            if self.stats['properties_found'] > 0:
                success_rate = ((self.stats['properties_saved'] + self.stats['properties_updated']) / 
                               self.stats['properties_found']) * 100
                report_data['summary']['success_rate'] = f"{success_rate:.1f}%"
            
            # Save report
            report_date = datetime.now().strftime('%Y%m%d')
            report_file = self.reports_dir / f"daily_report_{report_date}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Generated daily report: {report_file}")
            
            # Print summary to console
            self.print_final_report()
            
        except Exception as e:
            self.logger.error(f"❌ Error generating daily report: {e}")
    
    def send_email_report(self):
        """Send email report with automation results"""
        email_config = self.config['email']
        
        if not all([email_config.get('username'), email_config.get('password'), 
                   email_config.get('from_email'), email_config.get('to_emails')]):
            self.logger.info("📧 Email not configured, skipping email report")
            return
        
        try:
            duration = self.stats['end_time'] - self.stats['start_time']
            
            # Create email content
            subject = f"Tecnocasa Automation Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
Tecnocasa Automation Daily Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}

SUMMARY:
- Cities Processed: {self.stats['cities_processed']}/{len(self.cities)}
- Property Types: {len(self.property_types)}
- Pages Fetched: {self.stats['pages_fetched']}
- Properties Found: {self.stats['properties_found']}
- Properties Saved: {self.stats['properties_saved']}
- Properties Updated: {self.stats['properties_updated']}
- Properties Removed: {self.stats['properties_removed']}
- Errors: {self.stats['errors']}
- Blocked Requests: {self.stats['blocked_requests']}

"""
            
            if self.stats['properties_found'] > 0:
                success_rate = ((self.stats['properties_saved'] + self.stats['properties_updated']) / 
                               self.stats['properties_found']) * 100
                body += f"Success Rate: {success_rate:.1f}%\n\n"
            
            if self.stats['error_log']:
                body += "ERRORS:\n"
                for error in self.stats['error_log'][-10]:  # Last 10 errors
                    body += f"- {error}\n"
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            for to_email in email_config['to_emails']:
                server.send_message(msg, email_config['from_email'], to_email)
            
            server.quit()
            
            self.logger.info("📧 Email report sent successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending email report: {e}")
    
    def print_final_report(self):
        """Print final automation report to console"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("🏁 Automation Complete!")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Final Statistics:")
        self.logger.info(f"   Duration: {duration}")
        self.logger.info(f"   Cities processed: {self.stats['cities_processed']}/{len(self.cities)}")
        self.logger.info(f"   Property types: {len(self.property_types)}")
        self.logger.info(f"   Pages fetched: {self.stats['pages_fetched']}")
        self.logger.info(f"   Properties found: {self.stats['properties_found']}")
        self.logger.info(f"   Properties saved: {self.stats['properties_saved']}")
        self.logger.info(f"   Properties updated: {self.stats['properties_updated']}")
        self.logger.info(f"   Properties removed: {self.stats['properties_removed']}")
        self.logger.info(f"   Errors: {self.stats['errors']}")
        self.logger.info(f"   Blocked requests: {self.stats['blocked_requests']}")
        
        if self.stats['properties_found'] > 0:
            success_rate = ((self.stats['properties_saved'] + self.stats['properties_updated']) / 
                           self.stats['properties_found']) * 100
            self.logger.info(f"   Success rate: {success_rate:.1f}%")
        
        self.logger.info("=" * 80)
    
    def setup_scheduler(self):
        """Setup daily automation scheduler"""
        if not self.config['scheduling']['enable_scheduler']:
            self.logger.info("⏰ Scheduler disabled in configuration")
            return
        
        run_time = self.config['scheduling']['daily_run_time']
        run_on_weekends = self.config['scheduling']['run_on_weekends']
        
        self.logger.info(f"⏰ Setting up scheduler for daily run at {run_time}")
        
        if run_on_weekends:
            schedule.every().day.at(run_time).do(self.run_full_automation)
        else:
            schedule.every().monday.at(run_time).do(self.run_full_automation)
            schedule.every().tuesday.at(run_time).do(self.run_full_automation)
            schedule.every().wednesday.at(run_time).do(self.run_full_automation)
            schedule.every().thursday.at(run_time).do(self.run_full_automation)
            schedule.every().friday.at(run_time).do(self.run_full_automation)
        
        self.logger.info("⏰ Scheduler configured successfully")
    
    def run_scheduler(self):
        """Run the scheduler in a loop"""
        self.setup_scheduler()
        
        self.logger.info("🔄 Starting scheduler loop...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main entry point for production automation"""
    automation = ProductionTecnocasaAutomation()
    
    # Run immediately for testing
    automation.run_full_automation()


if __name__ == "__main__":
    main()
