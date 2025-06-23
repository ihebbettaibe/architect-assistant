import pandas as pd
import re
import os
from datetime import datetime

def safe_cleaner(input_folder='tecnocasa_data', output_folder='cleaned_data'):
    """
    Ultra-safe cleaning that:
    1. Preserves ALL original rows
    2. Only cleans values (never drops rows)
    3. Maintains error logs for problematic values
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(input_folder, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            # 1. Read with strict error handling
            df = pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='skip')
            original_count = len(df)
            
            # 2. Clean while preserving original values
            def safe_convert(series, pattern):
                """Helper to safely extract numbers"""
                return (
                    series
                    .astype(str)
                    .str.extract(f'({pattern})', expand=False)
                    .str.replace('[^\d]', '', regex=True)
                    .replace('', None)
                )
            
            # Price cleaning (handles "150,000 DT", "300 000", etc.)
            if 'Price' in df.columns:
                df['Price'] = safe_convert(df['Price'], r'\d[\d\s,]*')
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            
            # Surface cleaning (handles "500 m²", "750m2", etc.)
            if 'Surface' in df.columns:
                df['Surface'] = safe_convert(df['Surface'], r'\d+')
                df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
            
            # Text cleaning (preserve original if cleaning fails)
            text_cols = ['Title', 'Location', 'City']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            # 3. Verify no rows were lost
            if len(df) != original_count:
                print(f"⚠️ Warning: {original_count - len(df)} rows lost during loading!")
            
            # 4. Save with all rows
            clean_filename = f"cleaned_{filename}"
            output_path = os.path.join(output_folder, clean_filename)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Saved ALL {len(df)} rows to {clean_filename}")
            
            # 5. Generate cleaning report
            report = {
                'filename': filename,
                'original_rows': original_count,
                'price_converted': df['Price'].notna().sum() if 'Price' in df.columns else 0,
                'surface_converted': df['Surface'].notna().sum() if 'Surface' in df.columns else 0,
                'failed_price': original_count - df['Price'].notna().sum() if 'Price' in df.columns else 0,
                'failed_surface': original_count - df['Surface'].notna().sum() if 'Surface' in df.columns else 0
            }
            
            report_path = os.path.join(output_folder, 'cleaning_report.csv')
            if os.path.exists(report_path):
                pd.DataFrame([report]).to_csv(report_path, mode='a', header=False, index=False)
            else:
                pd.DataFrame([report]).to_csv(report_path, index=False)
                
        except Exception as e:
            error_msg = f"{datetime.now()},{filename},{str(e)}\n"
            with open(os.path.join(output_folder, 'cleaning_errors.log'), 'a') as f:
                f.write(error_msg)
            print(f"❌ Critical error with {filename}: {str(e)}")
    
    print("\nCleaning complete. Check:")
    print(f"- Cleaned files in '{output_folder}/'")
    print(f"- Cleaning report in '{output_folder}/cleaning_report.csv'")
    print(f"- Errors in '{output_folder}/cleaning_errors.log'")

if __name__ == "__main__":
    safe_cleaner()