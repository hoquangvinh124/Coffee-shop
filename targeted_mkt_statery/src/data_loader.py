"""
Data Loader Module for Targeted Marketing Strategy Project
Handles loading and initial processing of raw data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class DataLoader:
    """
    Class để load và merge dữ liệu từ 3 file JSON
    """
    
    def __init__(self, data_path='../data/'):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_path : str
            Đường dẫn đến thư mục data
        """
        self.data_path = Path(data_path)
        self.portfolio = None
        self.profile = None
        self.transcript = None
        self.merged_data = None
        
    def load_data(self):
        """Load 3 file JSON"""
        print("Loading data...")
        
        self.portfolio = pd.read_json(
            self.data_path / 'portfolio.json', 
            orient='records', 
            lines=True
        )
        self.profile = pd.read_json(
            self.data_path / 'profile.json', 
            orient='records', 
            lines=True
        )
        self.transcript = pd.read_json(
            self.data_path / 'transcript.json', 
            orient='records', 
            lines=True
        )
        
        print(f"✓ Portfolio: {self.portfolio.shape}")
        print(f"✓ Profile: {self.profile.shape}")
        print(f"✓ Transcript: {self.transcript.shape}")
        
        return self
    
    def clean_profile(self):
        """Xử lý missing values trong profile"""
        print("\nCleaning profile data...")
        before = len(self.profile)
        self.profile = self.profile.dropna()
        after = len(self.profile)
        
        print(f"✓ Dropped {before - after} rows with missing values")
        print(f"✓ Retained {after/before*100:.2f}% of data")
        
        return self
    
    def encode_customer_ids(self):
        """Encode customer IDs từ string sang integer"""
        print("\nEncoding customer IDs...")
        
        # Lấy unique customer IDs
        customer_ids = pd.unique(self.transcript['person'])
        customer_ids_dict = {cid: idx for idx, cid in enumerate(customer_ids)}
        
        # Map vào dataframes
        self.transcript['person'] = self.transcript['person'].map(customer_ids_dict)
        self.profile['id'] = self.profile['id'].map(customer_ids_dict)
        
        print(f"✓ Encoded {len(customer_ids_dict):,} customer IDs")
        
        return self
    
    def encode_offer_ids(self):
        """Encode offer IDs trong portfolio"""
        print("\nEncoding offer IDs...")
        
        offer_ids = self.portfolio['id'].unique()
        offer_ids_dict = {oid: idx for idx, oid in enumerate(offer_ids)}
        self.portfolio['id'] = self.portfolio['id'].map(offer_ids_dict)
        
        # Store mapping for later use
        self.offer_ids_dict = offer_ids_dict
        
        print(f"✓ Encoded {len(offer_ids_dict)} offer IDs")
        
        return self
    
    def merge_transcript_profile(self):
        """Merge transcript và profile"""
        print("\nMerging transcript and profile...")
        
        # Sort theo customer ID
        sorted_transcript = self.transcript.sort_values('person').reset_index(drop=True)
        sorted_profile = self.profile.sort_values('id').reset_index(drop=True)
        
        # Tính frequency của mỗi customer trong transcript
        customer_frequency = sorted_transcript['person'].value_counts(sort=False)
        sorted_profile = pd.concat([sorted_profile, customer_frequency], axis=1)
        
        # Duplicate profile rows dựa trên frequency
        profile_duplicated = sorted_profile.reindex(
            sorted_profile.index.repeat(sorted_profile['person'])
        ).reset_index(drop=True)
        
        # Drop column không cần thiết
        profile_duplicated = profile_duplicated.drop(['person'], axis=1)
        
        # Concatenate
        self.merged_data = pd.concat([sorted_transcript, profile_duplicated], axis=1)
        
        # Verify alignment
        assert (self.merged_data['person'] == self.merged_data['id']).all()
        
        # Drop duplicate id column
        self.merged_data = self.merged_data.drop(['person'], axis=1)
        
        print(f"✓ Merged data shape: {self.merged_data.shape}")
        
        return self
    
    def add_portfolio_features(self):
        """Thêm portfolio features vào merged data"""
        print("\nAdding portfolio features...")
        
        # Extract offer_id từ column 'value'
        def get_dict_value(x):
            if isinstance(x, dict):
                key = list(x.keys())[0]
                return x[key]
            return x
        
        offer_id_series = self.merged_data['value'].apply(get_dict_value)
        
        # Encode offer_ids
        def encode_offer_id(x):
            if isinstance(x, str):
                return self.offer_ids_dict.get(x, 10)
            else:
                return 10  # Transaction without offer
        
        self.merged_data['offer_id'] = offer_id_series.apply(encode_offer_id)
        
        # Add portfolio features
        portfolio_dict = self.portfolio.set_index('id')[['reward', 'difficulty', 'duration']].to_dict('index')
        
        self.merged_data['reward'] = self.merged_data['offer_id'].map(
            lambda x: portfolio_dict.get(x, {}).get('reward', 0)
        )
        self.merged_data['difficulty'] = self.merged_data['offer_id'].map(
            lambda x: portfolio_dict.get(x, {}).get('difficulty', 0)
        )
        self.merged_data['duration'] = self.merged_data['offer_id'].map(
            lambda x: portfolio_dict.get(x, {}).get('duration', 0)
        )
        
        print("✓ Added portfolio features: reward, difficulty, duration")
        
        return self
    
    def get_merged_data(self):
        """Trả về merged data"""
        if self.merged_data is None:
            raise ValueError("Data chưa được merge. Hãy chạy pipeline trước.")
        return self.merged_data
    
    def run_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("="*80)
        print("RUNNING DATA LOADER PIPELINE")
        print("="*80)
        
        (self.load_data()
             .clean_profile()
             .encode_customer_ids()
             .encode_offer_ids()
             .merge_transcript_profile()
             .add_portfolio_features())
        
        print("\n" + "="*80)
        print("✓ DATA LOADER PIPELINE COMPLETED")
        print("="*80)
        print(f"\nFinal merged data shape: {self.merged_data.shape}")
        print(f"Columns: {list(self.merged_data.columns)}")
        
        return self.merged_data


if __name__ == "__main__":
    # Test DataLoader
    loader = DataLoader(data_path='../data/')
    data = loader.run_pipeline()
    print("\n✓ DataLoader module test successful!")
    print(data.head())
