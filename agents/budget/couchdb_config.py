# CouchDB Configuration for Real Estate Budget Agent

# CouchDB Connection Settings
COUCHDB_URL = 'http://127.0.0.1:5984'
DB_NAME = 'realstate_budget'
USERNAME = 'admin'
PASSWORD = 'admin'

# Configuration options
USE_COUCHDB_BY_DEFAULT = True
FALLBACK_TO_CSV = True

# Query limits
DEFAULT_QUERY_LIMIT = 100
MAX_QUERY_LIMIT = 1000

# Cache settings
ENABLE_PROPERTY_CACHE = True
CACHE_DURATION_MINUTES = 30

# Data validation settings
MIN_VALID_PRICE = 1000        # Minimum valid property price (DT)
MAX_VALID_PRICE = 10000000    # Maximum valid property price (DT)
MIN_VALID_SURFACE = 10        # Minimum valid surface (m²)
MAX_VALID_SURFACE = 10000     # Maximum valid surface (m²)
