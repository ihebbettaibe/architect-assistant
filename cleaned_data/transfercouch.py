import csv
import json
import requests
import os

# CouchDB config
COUCHDB_URL = 'http://127.0.0.1:5984'
DB_NAME = 'realstate_budget'
USERNAME = 'admin'   # 🔁 Update if needed
PASSWORD = 'admin'   # 🔁 Update if needed

# Folder path containing your cleaned CSV files
FOLDER_PATH = r'C:\Users\ASUS\Desktop\architect-assistant\cleaned_data'

# List of your CSV files
CSV_FILES = [
    "cleaned_bizerte_properties.csv",
    "cleaned_grand_tunis_properties.csv",
    "cleaned_kairouan_properties.csv",
    "cleaned_mahdia_properties.csv",
    "cleaned_monastir_properties.csv",
    "cleaned_sfax_properties.csv",
    "cleaned_sousse_properties.csv"
]

# Create DB if not exists
def create_db():
    res = requests.put(f"{COUCHDB_URL}/{DB_NAME}", auth=(USERNAME, PASSWORD))
    if res.status_code == 201:
        print(f"✅ Created DB: {DB_NAME}")
    elif res.status_code == 412:
        print(f"ℹ️ DB already exists: {DB_NAME}")
    else:
        print(f"❌ Error creating DB: {res.text}")

# Upload one CSV file
def upload_csv_to_couch(csv_file_path, doc_counter_start=0):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = doc_counter_start
        for row in reader:
            doc = {
                "_id": f"{os.path.basename(csv_file_path).replace('.csv', '')}_{count}",
                "city": row.get("City", ""),
                "title": row.get("Title", ""),
                "type": row.get("Type", ""),
                "price": row.get("Price", ""),
                "surface": row.get("Surface", ""),
                "location": row.get("Location", ""),
                "url": row.get("URL", "")
            }

            res = requests.put(
                f"{COUCHDB_URL}/{DB_NAME}/{doc['_id']}",
                auth=(USERNAME, PASSWORD),
                headers={"Content-Type": "application/json"},
                data=json.dumps(doc)
            )

            if res.status_code in [201, 202]:
                print(f"✅ Uploaded: {doc['_id']}")
            else:
                print(f"⚠️ Failed {doc['_id']}: {res.text}")
            count += 1

# Upload all CSVs
create_db()
counter = 1
for file in CSV_FILES:
    full_path = os.path.join(FOLDER_PATH, file)
    print(f"\n📤 Processing: {file}")
    upload_csv_to_couch(full_path, doc_counter_start=counter)
    counter += 10000  # Prevent ID collisions
