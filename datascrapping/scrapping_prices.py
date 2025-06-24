import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import json
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
    'bizerte': {'slug': 'nord-est-ne/bizerte', 'name': 'Bizerte'}
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Create output directories
os.makedirs('tecnocasa_data', exist_ok=True)

def scrape_city(city_data):
    """Scrape ALL listings for a city across all pages"""
    city_slug = city_data['slug']
    city_name = city_data['name']
    page = 1
    all_listings = []
    max_pages = 20  # Safety limit
    session = requests.Session()
    
    while page <= max_pages:
        url = f"{BASE_URL}/vendre/terrain/{city_slug}.html?page={page}"
        print(f"Scraping {city_name} - Page {page}...")
        
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
                    time.sleep(5)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract JSON data from the estates-index tag
            estates_tag = soup.find("estates-index")
            if not estates_tag:
                print(f"No listings found on page {page} - stopping.")
                break
                
            estates_json = estates_tag.get(":estates", "").replace('&quot;', '"')
            try:
                listings = json.loads(estates_json)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON on page {page}: {str(e)}")
                break
                
            if not listings:
                print(f"No more listings found - reached end of results.")
                break
                
            # Process listings
            for estate in listings:
                all_listings.append({
                    "City": city_name,
                    "Title": estate.get("title", "").strip(),
                    "Price": estate.get("price", "").replace("DT", "").replace(",", "").strip(),
                    "Surface": estate.get("surface", "").replace("m²", "").strip(),
                    "Location": estate.get("subtitle", "").strip(),
                    "URL": urljoin(BASE_URL, estate.get("detail_url", ""))
                })
            
            page += 1
            time.sleep(2 + (page % 5))  # Randomized delay
            
        except Exception as e:
            print(f"Error scraping page {page}: {str(e)}")
            break
    
    # Save results
    if all_listings:
        df = pd.DataFrame(all_listings)
        # Convert price and surface to numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
        
        # Generate filename
        clean_city_name = city_name.lower().replace(" ", "_")
        filename = f"tecnocasa_data/{clean_city_name}_properties.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Saved {len(df)} listings to {filename}\n")
        return filename
    else:
        print(f"❌ No listings found for {city_name}\n")
        return None

def main():
    print(f"🚀 Starting Tecnocasa Scraper for {len(CITIES)} regions")
    start_time = time.time()
    
    results = {}
    for city_key, city_data in CITIES.items():
        results[city_key] = scrape_city(city_data)
    
    total_time = time.time() - start_time
    print(f"🏁 Completed in {total_time:.2f} seconds")
    return results

if __name__ == "__main__":
    main()