import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import json
import glob
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://www.tecnocasa.tn"
CITIES = {
    'grand-tunis': {'slug': 'nord-est-ne/grand-tunis', 'name': 'Grand Tunis'},
    'sousse': {'slug': 'centre-est-ce/sousse', 'name': 'Sousse'},
    'sfax': {'slug': 'centre-est-ce/sfax', 'name': 'Sfax'},
    'monastir': {'slug': 'centre-est-ce/monastir', 'name': 'Monastir'},
    'mahdia': {'slug': 'centre-est-ce/mahdia', 'name': 'Mahdia'},
    'kairouan': {'slug': 'centre-est-ce/kairouan', 'name': 'Kairouan'},
    'bizerte': {'slug': 'nord-est-ne/bizerte', 'name': 'Bizerte'},
    'cap-bon': {'slug': 'centre-est-ce/cap-bon', 'name': 'Cap Bon'}
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.tecnocasa.tn/',
}

def create_directories():
    """Create necessary directories for saving data"""
    os.makedirs('tecnocasa_data', exist_ok=True)
    os.makedirs('tecnocasa_html', exist_ok=True)

def save_html_backup(city_name, html_content):
    """Save HTML backup for debugging purposes"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clean_city_name = city_name.lower().replace(" ", "_")
    filename = f"tecnocasa_html/{clean_city_name}_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML backup saved: {filename}")
    return filename

def scrape_city(city_key, city_data, save_html=False):
    """Scrape ALL listings for a city across all pages"""
    city_slug = city_data['slug']
    city_name = city_data['name']
    page = 1
    all_listings = []
    max_pages = 50  # Increased safety limit
    session = requests.Session()
    
    print(f"🏙️ Starting scrape for {city_name}...")
    
    while page <= max_pages:
        url = f"{BASE_URL}/vendre/terrain/{city_slug}.html?page={page}"
        print(f"   📄 Scraping page {page}...")
        
        try:
            # Fetch page with retry logic
            for attempt in range(3):
                try:
                    response = session.get(url, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                    
                    # Check for CAPTCHA or blocking
                    if "captcha" in response.text.lower():
                        print("⚠️ CAPTCHA detected! Try again later or use proxies")
                        return None
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        raise
                    print(f"   ⏳ Retry {attempt + 1}/3 for page {page}")
                    time.sleep(5)
            
            # Save HTML backup for first page if requested
            if page == 1 and save_html:
                save_html_backup(city_name, response.text)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract JSON data from the estates-index tag
            estates_tag = soup.find("estates-index")
            if not estates_tag:
                print(f"   ❌ No listings found on page {page} - stopping.")
                break
                
            estates_json = estates_tag.get(":estates", "").replace('&quot;', '"')
            try:
                listings = json.loads(estates_json)
            except json.JSONDecodeError as e:
                print(f"   ❌ Failed to parse JSON on page {page}: {str(e)}")
                break
                
            if not listings:
                print(f"   ✅ No more listings found - reached end of results.")
                break
                
            # Process listings
            for estate in listings:
                listing_data = {
                    "City": city_name,
                    "Title": estate.get("title", "").strip(),
                    "Subtitle": estate.get("subtitle", "").strip(),
                    "Price": estate.get("price", "").replace("DT", "").replace(",", "").strip(),
                    "Previous_Price": estate.get("previous_price", ""),
                    "Discount": estate.get("discount", ""),
                    "Surface": estate.get("surface", "").replace("m²", "").strip(),
                    "Location": estate.get("subtitle", "").strip(),
                    "URL": urljoin(BASE_URL, estate.get("detail_url", "")),
                    "Page": page
                }
                all_listings.append(listing_data)
            
            print(f"   ✅ Found {len(listings)} listings on page {page}")
            page += 1
            
            # Dynamic delay to avoid rate limiting
            delay = 2 + (page % 5) + (0.5 if page > 10 else 0)
            time.sleep(delay)
            
        except Exception as e:
            print(f"   ❌ Error scraping page {page}: {str(e)}")
            break
    
    # Save results
    if all_listings:
        df = pd.DataFrame(all_listings)
        
        # Clean and convert numeric columns
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
        
        # Generate filename
        clean_city_name = city_name.lower().replace(" ", "_")
        filename = f"tecnocasa_data/{clean_city_name}_properties.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Saved {len(df)} listings for {city_name} to {filename}")
        return filename, len(df)
    else:
        print(f"❌ No listings found for {city_name}")
        return None, 0

def check_existing_data(city_name):
    """Check if data already exists for a city"""
    clean_city_name = city_name.lower().replace(" ", "_")
    pattern = f"tecnocasa_data/{clean_city_name}_properties.csv"
    existing_files = glob.glob(pattern)
    
    if existing_files:
        # Get the most recent file
        latest_file = max(existing_files, key=os.path.getctime)
        file_age = time.time() - os.path.getctime(latest_file)
        
        # If file is less than 24 hours old, consider it fresh
        if file_age < 24 * 3600:
            df = pd.read_csv(latest_file)
            print(f"📊 Found recent data for {city_name}: {len(df)} listings (file age: {file_age/3600:.1f}h)")
            return True, len(df)
    
    return False, 0

def consolidate_all_data():
    """Consolidate all city data into one master CSV"""
    all_files = glob.glob("tecnocasa_data/*_properties.csv")
    
    if not all_files:
        print("❌ No data files found to consolidate")
        return
    
    all_data = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"📊 Loaded {len(df)} records from {os.path.basename(file)}")
        except Exception as e:
            print(f"❌ Error loading {file}: {str(e)}")
    
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_filename = "tecnocasa_data/all_tunisia_properties.csv"
        master_df.to_csv(master_filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Consolidated {len(master_df)} total listings into {master_filename}")
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        print(f"   Total listings: {len(master_df)}")
        print(f"   Cities covered: {master_df['City'].nunique()}")
        print(f"   Price range: {master_df['Price'].min():.0f} - {master_df['Price'].max():.0f} DT")
        print(f"   Surface range: {master_df['Surface'].min():.0f} - {master_df['Surface'].max():.0f} m²")
        print("\n🏙️ Listings by city:")
        city_counts = master_df['City'].value_counts()
        for city, count in city_counts.items():
            print(f"   {city}: {count} listings")
        
        return master_filename
    else:
        print("❌ No data to consolidate")
        return None

def main(cities_to_scrape=None, force_scrape=False, save_html=False):
    """
    Main scraping function
    
    Args:
        cities_to_scrape: List of city keys to scrape (None for all)
        force_scrape: Force scrape even if recent data exists
        save_html: Save HTML backups for debugging
    """
    print(f"🚀 Starting Tecnocasa Tunisia Scraper")
    print(f"   Target cities: {cities_to_scrape or 'ALL'}")
    print(f"   Force scrape: {force_scrape}")
    print(f"   Save HTML: {save_html}")
    
    create_directories()
    start_time = time.time()
    
    # Determine which cities to scrape
    if cities_to_scrape:
        cities_to_process = {k: v for k, v in CITIES.items() if k in cities_to_scrape}
    else:
        cities_to_process = CITIES
    
    results = {}
    total_listings = 0
    
    for city_key, city_data in cities_to_process.items():
        city_name = city_data['name']
        
        # Check if we should skip this city
        if not force_scrape:
            has_recent_data, existing_count = check_existing_data(city_name)
            if has_recent_data:
                print(f"⏭️ Skipping {city_name} (recent data exists)")
                results[city_key] = {'status': 'skipped', 'count': existing_count}
                total_listings += existing_count
                continue
        
        # Scrape the city
        try:
            filename, count = scrape_city(city_key, city_data, save_html)
            if filename:
                results[city_key] = {'status': 'success', 'file': filename, 'count': count}
                total_listings += count
            else:
                results[city_key] = {'status': 'failed', 'count': 0}
        except Exception as e:
            print(f"❌ Failed to scrape {city_name}: {str(e)}")
            results[city_key] = {'status': 'error', 'count': 0, 'error': str(e)}
    
    # Consolidate all data
    master_file = consolidate_all_data()
    
    total_time = time.time() - start_time
    print(f"\n🏁 Scraping completed in {total_time:.2f} seconds")
    print(f"📊 Total listings processed: {total_listings}")
    
    # Print final summary
    print("\n📋 Final Results:")
    for city_key, result in results.items():
        city_name = CITIES[city_key]['name']
        status = result['status'].upper()
        count = result['count']
        print(f"   {city_name:<15}: {status:<10} ({count} listings)")
    if master_file:
        master_df = pd.read_csv(master_file)    
        json_output = master_df.to_dict(orient="records")
        print(json.dumps(json_output, ensure_ascii=False))  

    return results

if __name__ == "__main__":
    # Example usage:
    # Scrape all cities
    results = main()
    
    # Scrape only specific cities
    # results = main(cities_to_scrape=['grand-tunis', 'sousse'])
    
    # Force scrape even if recent data exists
    # results = main(force_scrape=True)
    
    # Save HTML backups for debugging
    # results = main(save_html=True)