#!/usr/bin/env python3
"""
Script to fix missing prices in cleaned CSV files
"""

import pandas as pd
import os

def fix_missing_prices():
    """Apply price estimation to all CSV files with missing prices"""
    
    data_folder = "cleaned_data"
    
    # City-specific price defaults per m²
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
    
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            filepath = os.path.join(data_folder, file)
            print(f"\n🔧 Processing {file}...")
            
            try:
                # Load the CSV
                df = pd.read_csv(filepath)
                
                # Check for required columns
                if 'Price' not in df.columns or 'Surface' not in df.columns:
                    print(f"⚠️ Skipping {file}: missing required columns")
                    continue
                
                # Convert to numeric
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
                
                # Remove invalid surface data
                initial_count = len(df)
                df = df.dropna(subset=['Surface'])
                df = df[df['Surface'] > 0]
                print(f"   📊 Valid surface data: {len(df)}/{initial_count}")
                
                # Find missing prices
                missing_price_mask = df['Price'].isna() | (df['Price'] <= 0)
                missing_count = missing_price_mask.sum()
                
                if missing_count > 0:
                    print(f"   ⚠️ Found {missing_count} properties with missing prices")
                    
                    # Try to get average from valid data first
                    valid_prices = df[~missing_price_mask]
                    
                    if len(valid_prices) > 0:
                        avg_price_per_m2 = (valid_prices['Price'] / valid_prices['Surface']).mean()
                        print(f"   📊 Average price/m² from valid data: {avg_price_per_m2:.0f} TND/m²")
                        
                        if not pd.isna(avg_price_per_m2):
                            # Use calculated average
                            df.loc[missing_price_mask, 'Price'] = df.loc[missing_price_mask, 'Surface'] * avg_price_per_m2
                            print(f"   ✅ Estimated prices using calculated average")
                        else:
                            # Use city defaults
                            city_name = df['City'].iloc[0] if len(df) > 0 else 'Unknown'
                            default_price = city_defaults.get(city_name, 1500)  # Default fallback
                            df.loc[missing_price_mask, 'Price'] = df.loc[missing_price_mask, 'Surface'] * default_price
                            print(f"   ✅ Estimated prices using city default ({default_price} TND/m²)")
                    else:
                        # Use city defaults
                        city_name = df['City'].iloc[0] if len(df) > 0 else 'Unknown'
                        default_price = city_defaults.get(city_name, 1500)
                        df.loc[missing_price_mask, 'Price'] = df.loc[missing_price_mask, 'Surface'] * default_price
                        print(f"   ✅ Estimated prices using city default ({default_price} TND/m²)")
                
                # Remove any remaining invalid data
                df = df.dropna(subset=['Price'])
                df = df[df['Price'] > 0]
                
                # Add calculated fields
                df['price_per_m2'] = df['Price'] / df['Surface']
                
                print(f"   📊 Final count: {len(df)} properties")
                print(f"   💰 Price range: {df['Price'].min():,.0f} - {df['Price'].max():,.0f} TND")
                print(f"   📊 Average price: {df['Price'].mean():,.0f} TND")
                
                # Save the updated file
                df.to_csv(filepath, index=False)
                print(f"   ✅ Updated {file}")
                
            except Exception as e:
                print(f"   ❌ Error processing {file}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing missing prices in CSV files...")
    fix_missing_prices()
    print("\n✅ Price estimation complete!")
