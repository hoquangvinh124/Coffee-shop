"""
Preprocessor Module for Targeted Marketing Strategy Project
Handles feature engineering, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    """
    Class để preprocessing dữ liệu
    """
    
    def __init__(self, data):
        """
        Initialize Preprocessor
        
        Parameters:
        -----------
        data : DataFrame
            Merged data từ DataLoader
        """
        self.data = data.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Mappings
        self.event_mapping = {
            'offer received': 0,
            'offer viewed': 1,
            'transaction': 2,
            'offer completed': 3,
            'green flag': 4
        }
        
        self.gender_mapping = {'F': 0, 'M': 1, 'O': 2}
        
    def extract_registration_month(self):
        """Extract month từ became_member_on"""
        print("Extracting registration month...")
        
        def extract_month(date_int):
            date_str = str(date_int)
            if len(date_str) == 8:
                return int(date_str[4:6])
            return 0
        
        self.data['reg_month'] = self.data['became_member_on'].apply(extract_month)
        print("✓ Registration month extracted")
        
        return self
    
    def create_target_variable(self):
        """Tạo target variable (event_id)"""
        print("Creating target variable...")
        
        self.data['event_id'] = self.data['event'].map(self.event_mapping)
        print(f"✓ Target variable created")
        print(f"Event distribution:\n{self.data['event_id'].value_counts().sort_index()}")
        
        return self
    
    def drop_unnecessary_columns(self):
        """Drop các columns không cần thiết"""
        print("\nDropping unnecessary columns...")
        
        columns_to_drop = ['event', 'value', 'time', 'became_member_on']
        self.data = self.data.drop(columns=columns_to_drop)
        
        print(f"✓ Dropped {len(columns_to_drop)} columns")
        print(f"Remaining columns: {list(self.data.columns)}")
        
        return self
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        print("\nEncoding categorical features...")
        
        # Encode gender
        self.data['gender'] = self.data['gender'].map(self.gender_mapping)
        
        print("✓ Encoded gender: F/M/O → 0/1/2")
        print(f"Gender distribution: {self.data['gender'].value_counts().sort_index().to_dict()}")
        
        return self
    
    def split_features_target(self):
        """Tách features và target"""
        print("\nSplitting features and target...")
        
        X = self.data.drop(['id', 'event_id'], axis=1)
        y = self.data['event_id']
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        
        return X, y
    
    def train_test_split(self, X, y, test_size=0.25, random_state=42):
        """Split thành train và test sets"""
        print("\nSplitting train/test sets...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Reset index
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        
        print(f"✓ Train set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        
        return self
    
    def scale_features(self):
        """Scale features"""
        print("\nScaling features...")
        
        # StandardScaler cho age và income
        std_scaler = StandardScaler()
        
        for col in ['age', 'income']:
            self.X_train[col] = std_scaler.fit_transform(self.X_train[[col]])
            self.X_test[col] = std_scaler.transform(self.X_test[[col]])
        
        print("✓ StandardScaler applied to: age, income")
        
        # MinMaxScaler cho reward, difficulty, reg_month
        minmax_scaler = MinMaxScaler()
        
        for col in ['reward', 'difficulty', 'reg_month']:
            self.X_train[col] = minmax_scaler.fit_transform(self.X_train[[col]])
            self.X_test[col] = minmax_scaler.transform(self.X_test[[col]])
        
        print("✓ MinMaxScaler applied to: reward, difficulty, reg_month")
        
        return self
    
    def apply_oversampling(self, strategy='not majority'):
        """Apply random oversampling để handle imbalanced data"""
        print("\nApplying Random Oversampling...")
        
        print(f"Before oversampling - Train shape: {self.X_train.shape}")
        print(f"Class distribution:\n{self.y_train.value_counts().sort_index()}")
        
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)
        
        print(f"\n✓ After oversampling - Train shape: {self.X_train.shape}")
        print(f"Class distribution:\n{self.y_train.value_counts().sort_index()}")
        
        return self
    
    def get_train_test_data(self):
        """Trả về train/test data"""
        if self.X_train is None:
            raise ValueError("Data chưa được split. Hãy chạy pipeline trước.")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def run_pipeline(self, apply_oversampling=False):
        """Chạy toàn bộ preprocessing pipeline"""
        print("="*80)
        print("RUNNING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Feature engineering
        (self.extract_registration_month()
             .create_target_variable()
             .drop_unnecessary_columns()
             .encode_categorical_features())
        
        # Split features and target
        X, y = self.split_features_target()
        
        # Train/test split và scaling
        (self.train_test_split(X, y)
             .scale_features())
        
        # Optional: Apply oversampling
        if apply_oversampling:
            self.apply_oversampling()
        
        print("\n" + "="*80)
        print("✓ PREPROCESSING PIPELINE COMPLETED")
        print("="*80)
        print(f"\nFinal shapes:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
        
        return self.get_train_test_data()
    
    def save_processed_data(self, output_path='../data/processed/'):
        """Lưu processed data"""
        import os
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu datasets
        self.X_train.to_csv(output_path / 'X_train.csv', index=False)
        self.X_test.to_csv(output_path / 'X_test.csv', index=False)
        self.y_train.to_csv(output_path / 'y_train.csv', index=False)
        self.y_test.to_csv(output_path / 'y_test.csv', index=False)
        
        # Lưu metadata
        import pickle
        metadata = {
            'feature_names': list(self.X_train.columns),
            'event_mapping': self.event_mapping,
            'gender_mapping': self.gender_mapping,
            'train_shape': self.X_train.shape,
            'test_shape': self.X_test.shape
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Processed data saved to: {output_path}")


if __name__ == "__main__":
    # Test Preprocessor
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(data_path='../data/')
    merged_data = loader.run_pipeline()
    
    # Preprocess
    preprocessor = Preprocessor(merged_data)
    X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
    
    print("\n✓ Preprocessor module test successful!")
