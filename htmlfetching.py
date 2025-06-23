import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin
import glob
import json
import pandas as pd

# Configuration
BASE_URL = "https://www.tecnocasa.tn"
CITIES = {
    'cap-bon': 'cap-bon',
    'bizerte': 'bizerte',
    'grand-tunis': 'grand-tunis',
    'kairouan': 'kairouan',
    'mahdia': 'mahdia',
    'monastir': 'monastir',
    'sfax': 'sfax',
    'sousse': 'sousse'
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.tecnocasa.tn/',
}

def create_directory():
    """Create directory for saving HTML files if it doesn't exist"""
    if not os.path.exists('tecnocasa_html'):
        os.makedirs('tecnocasa_html')

def scrape_city(city_name, city_slug):
    """Scrape listings for a specific city and save HTML"""
    url = f"{BASE_URL}/vendre/terrain/centre-est-ce/{city_slug}.html"
    print(f"\nScraping {city_name} ({url})...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Save raw HTML
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tecnocasa_html/{city_name.lower().replace(' ', '_')}_{timestamp}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        
        print(f"Successfully saved {city_name} data to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error scraping {city_name}: {str(e)}")
        return None

def scrape_all_cities():
    """Scrape only grand-tunis if not already scraped, skip others."""
    create_directory()
    results = {}

    for city_name, city_slug in CITIES.items():
        # Only process grand-tunis
        if city_slug != "grand-tunis":
            print(f"Skipping {city_name} (already scraped or not required).")
            continue

        # Check if file already exists
        already_scraped = False
        for fname in os.listdir('tecnocasa_html'):
            if fname.startswith(city_name.lower().replace(' ', '_')) and fname.endswith('.html'):
                print(f"Already scraped {city_name}, skipping.")
                already_scraped = True
                results[city_name] = os.path.join('tecnocasa_html', fname)
                break

        if already_scraped:
            continue

        html_file = scrape_city(city_name, city_slug)
        results[city_name] = html_file
        time.sleep(5)  # Be polite

    return results

def extract_grand_tunis_to_csv():
    # Find the latest grand-tunis html file
    files = sorted(glob.glob("tecnocasa_html/grand-tunis_*.html"), reverse=True)
    if not files:
        print("No grand-tunis HTML file found.")
        return
    html_file = files[0]
    print(f"Extracting data from {html_file}")

    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    estates_tag = soup.find("estates-index")
    if not estates_tag:
        print("No <estates-index> tag found in HTML.")
        return

    estates_json = estates_tag.get(":estates")
    if not estates_json:
        print("No ':estates' attribute found in <estates-index> tag.")
        return

    estates_json = estates_json.replace('&quot;', '"')
    try:
        estates = json.loads(estates_json)
    except Exception as e:
        print(f"JSON decode error: {e}")
        return

    data = []
    for estate in estates:
        data.append({
            "Title": estate.get("title"),
            "Subtitle": estate.get("subtitle"),
            "Surface": estate.get("surface"),
            "Price": estate.get("price"),
            "Previous Price": estate.get("previous_price"),
            "Discount": estate.get("discount"),
            "Location": estate.get("subtitle"),
            "URL": estate.get("detail_url"),
        })

    df = pd.DataFrame(data)
    out_csv = "tecnocasa_grand_tunis.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} estates to {out_csv}")

if __name__ == "__main__":
    print("Starting Tecnocasa Tunisia scraper...")
    scraped_files = scrape_all_cities()
    
    print("\nScraping completed. Results:")
    for city, filepath in scraped_files.items():
        status = "SUCCESS" if filepath else "FAILED"
        print(f"{city.upper().replace('-', ' '):<12} : {status} - {filepath or 'No file saved'}")

    # Extract grand-tunis HTML to CSV
    extract_grand_tunis_to_csv()