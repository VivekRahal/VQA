"""
Data preparation script for VQA system.

This script prepares the dataset by splitting it into train/validation/test sets
and creating corresponding image list files for efficient data loading.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split


class DataPreparator:
    """
    Data preparation class for VQA dataset.
    
    This class handles splitting the dataset into train/validation/test sets
    and creating corresponding image list files for efficient data loading.
    
    Attributes:
        data_dir (str): Directory containing the dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
    """
    
    def __init__(self, 
                 data_dir: str = "data", 
                 train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, 
                 test_ratio: float = 0.1) -> None:
        """
        Initialize the data preparator.
        
        Args:
            data_dir: Directory containing the dataset
            train_ratio: Ratio of training data (0.0 to 1.0)
            val_ratio: Ratio of validation data (0.0 to 1.0)
            test_ratio: Ratio of test data (0.0 to 1.0)
            
        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        if not (0.0 <= train_ratio <= 1.0 and 0.0 <= val_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0):
            raise ValueError("All ratios must be between 0.0 and 1.0")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is empty or malformed
        """
        file_path = os.path.join(self.data_dir, csv_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        
        required_columns = ['image_id', 'question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Loaded {len(df)} samples from {csv_file}")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train + temp, test
        train_temp, test_df = train_test_split(
            df, 
            test_size=self.test_ratio, 
            random_state=42, 
            stratify=df['answer'] if 'answer' in df.columns else None
        )
        
        # Second split: train, validation
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_temp, 
            test_size=val_ratio_adjusted, 
            random_state=42, 
            stratify=train_temp['answer'] if 'answer' in train_temp.columns else None
        )
        
        print(f"✅ Data split completed:")
        print(f"   Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_split_data(self, 
                       train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> None:
        """
        Save split data to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
        """
        # Save CSV files
        train_df.to_csv(os.path.join(self.data_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(self.data_dir, 'test_split.csv'), index=False)
        
        print("✅ Split data saved to CSV files")
    
    def create_image_lists(self, 
                          train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> None:
        """
        Create image list files for each split.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
        """
        # Get unique image IDs for each split
        train_images = sorted(train_df['image_id'].unique())
        val_images = sorted(val_df['image_id'].unique())
        test_images = sorted(test_df['image_id'].unique())
        
        # Save image lists
        self._save_image_list(train_images, 'train_images_split.txt')
        self._save_image_list(val_images, 'val_images_split.txt')
        self._save_image_list(test_images, 'test_images_split.txt')
        
        print("✅ Image list files created")
    
    def _save_image_list(self, image_ids: List[str], filename: str) -> None:
        """
        Save image IDs to a text file.
        
        Args:
            image_ids: List of image IDs
            filename: Output filename
        """
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, 'w') as f:
            for image_id in image_ids:
                f.write(f"{image_id}\n")
    
    def prepare_dataset(self, csv_file: str = "data_train.csv") -> None:
        """
        Prepare the complete dataset by splitting and creating necessary files.
        
        Args:
            csv_file: Name of the input CSV file
            
        Raises:
            ValueError: If preparation fails
        """
        print("🔄 Starting dataset preparation...")
        print("=" * 50)
        
        try:
            # Load data
            df = self.load_data(csv_file)
            
            # Split data
            train_df, val_df, test_df = self.split_data(df)
            
            # Save split data
            self.save_split_data(train_df, val_df, test_df)
            
            # Create image lists
            self.create_image_lists(train_df, val_df, test_df)
            
            print("\n✅ Dataset preparation completed successfully!")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n❌ Dataset preparation failed: {e}")
            raise ValueError(f"Dataset preparation failed: {e}")


def main() -> None:
    """Main function to prepare the dataset."""
    try:
        # Create data preparator
        preparator = DataPreparator(
            data_dir="data",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Prepare dataset
        preparator.prepare_dataset("data_train.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 