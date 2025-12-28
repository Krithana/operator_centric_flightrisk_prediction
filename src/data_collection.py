# src/data_collection.py - ENCODING FIXED VERSION
import pandas as pd
import os
import glob
import chardet

class EncodingFixedDataLoader:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_path = os.path.join(self.project_root, "data", "raw")
        self.processed_path = os.path.join(self.project_root, "data", "processed")
        
        print(f"🔧 PROJECT PATHS:")
        print(f"   Project root: {self.project_root}")
        print(f"   Raw data path: {self.raw_path}")
        
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def detect_encoding(self, file_path):
        """Detect file encoding automatically"""
        print(f"🔍 Detecting encoding for: {os.path.basename(file_path)}")
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB to detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"   Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    
    def load_with_encoding(self, file_path, encodings_to_try=None):
        """Load CSV file trying multiple encodings"""
        if encodings_to_try is None:
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                print(f"   Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"   ✅ SUCCESS with {encoding} encoding!")
                return df
            except UnicodeDecodeError as e:
                print(f"   ❌ Failed with {encoding}: {e}")
                continue
            except Exception as e:
                print(f"   ❌ Other error with {encoding}: {e}")
                continue
        
        # If automatic detection fails, try detecting encoding
        try:
            detected_encoding = self.detect_encoding(file_path)
            if detected_encoding and detected_encoding not in encodings_to_try:
                print(f"   🎯 Trying detected encoding: {detected_encoding}")
                df = pd.read_csv(file_path, encoding=detected_encoding, low_memory=False)
                print(f"   ✅ SUCCESS with detected encoding {detected_encoding}!")
                return df
        except Exception as e:
            print(f"   ❌ Failed with detected encoding: {e}")
        
        return None
    
    def load_datasets(self):
        """Load both datasets with encoding handling"""
        print("\n📂 LOADING DATASETS WITH ENCODING FIX...")
        
        datasets = {}
        
        # List all files
        all_files = os.listdir(self.raw_path)
        print(f"📁 Files found in data/raw: {len(all_files)}")
        for file in all_files:
            print(f"   📄 {file}")
        
        # 1. Load Aviation Accident Data
        accidents_file = os.path.join(self.raw_path, "AviationData.csv")
        if os.path.exists(accidents_file):
            print(f"\n🚀 LOADING AVIATION ACCIDENT DATA...")
            df_accidents = self.load_with_encoding(accidents_file)
            if df_accidents is not None:
                datasets['accidents'] = df_accidents
                print(f"✅ SUCCESS: Loaded accidents data - {df_accidents.shape}")
            else:
                print("❌ FAILED: Could not load accidents data with any encoding")
        else:
            print("❌ AviationData.csv not found!")
        
        # 2. Load Airline Safety Data (this one works fine)
        safety_file = os.path.join(self.raw_path, "airline-safety.csv")
        if os.path.exists(safety_file):
            print(f"\n🚀 LOADING AIRLINE SAFETY DATA...")
            try:
                df_safety = pd.read_csv(safety_file)
                datasets['safety'] = df_safety
                print(f"✅ SUCCESS: Loaded safety data - {df_safety.shape}")
            except Exception as e:
                print(f"❌ Failed to load safety data: {e}")
        else:
            print("❌ airline-safety.csv not found!")
        
        return datasets
    
    def validate_and_save(self, datasets):
        """Validate datasets and save processed versions"""
        print("\n🔍 VALIDATING AND SAVING DATASETS...")
        
        if 'accidents' in datasets:
            df = datasets['accidents']
            print(f"📊 ACCIDENTS DATASET:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Column names: {list(df.columns)}")
            print(f"   Sample data (first 2 rows):")
            print(df.head(2).to_string())
            
            # Save processed version
            output_path = os.path.join(self.processed_path, "accidents_processed.csv")
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"   💾 Saved: {output_path}")
        
        if 'safety' in datasets:
            df = datasets['safety']
            print(f"\n📊 SAFETY DATASET:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Airlines: {df['airline'].nunique()}")
            print(f"   Sample data (first 2 rows):")
            print(df.head(2).to_string())
            
            # Save processed version
            output_path = os.path.join(self.processed_path, "safety_processed.csv")
            df.to_csv(output_path, index=False)
            print(f"   💾 Saved: {output_path}")
        
        return len(datasets)

def main():
    loader = EncodingFixedDataLoader()
    
    print("🎯 ENCODING-FIXED DATA LOADER")
    print("=" * 60)
    print("Fixing encoding issues in AviationData.csv...")
    print("=" * 60)
    
    # Load datasets
    datasets = loader.load_datasets()
    
    # Validate and save
    loaded_count = loader.validate_and_save(datasets)
    
    print(f"\n{'🎉' if loaded_count == 2 else '⚠️'} FINAL RESULT:")
    print(f"   Successfully loaded {loaded_count}/2 datasets")
    
    if loaded_count == 2:
        print("\n✅ ALL RESEARCH DATA READY!")
        print("   Next: Begin analysis in notebooks/01_real_data_exploration.ipynb")
    else:
        print("\n❌ Some datasets failed to load.")
        if 'accidents' not in datasets:
            print("   - Aviation Accident Data (encoding issue)")
        if 'safety' not in datasets:
            print("   - Airline Safety Data")

if __name__ == "__main__":
    main()