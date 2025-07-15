"""
Data Processing Module for Incident Classification
Handles data loading, cleaning, and preparation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import yaml
from pathlib import Path

class DataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from CSV file"""
        if file_path is None:
            file_path = self.config['data']['raw_file']
        return pd.read_csv(file_path, encoding='utf-8')
    
    def prepare_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and concatenate text fields for embedding"""
        df = df.copy()
        
        # Create raw text combination (original fields)
        df['raw_text'] = (
            df['Service'].fillna('') + ' | ' +
            df['Category'].fillna('') + ' | ' +
            df['SubCategory'].fillna('') + ' | ' +
            df['SubCategory_Prefix '].fillna('') + ' | ' +
            df['SubCategory_Keywords'].fillna('') + ' | ' +
            df['SubCategory2'].fillna('') + ' | ' +
            df['SubCategory2_Prefix '].fillna('') + ' | ' +
            df['SubCategory2_Keywords'].fillna('')
        )
        
        # Create structured text for AI agent (original fields)
        df['structured_text'] = df.apply(self._create_structured_text, axis=1)
        
        # For user queries - we'll use a simplified approach
        df['user_query_format'] = (
            df['SubCategory'].fillna('') + ' ' +
            df['SubCategory2'].fillna('')
        ).str.strip()
        
        return df
    
    def _create_structured_text(self, row) -> str:
        """Create structured text for AI agent processing"""
        return f"""
        Service: {row.get('Service', '')}
        Category: {row.get('Category', '')}
        SubCategory: {row.get('SubCategory', '')}
        SubCategory_Prefix: {row.get('SubCategory_Prefix ', '')}
        SubCategory_Keywords: {row.get('SubCategory_Keywords', '')}
        SubCategory2: {row.get('SubCategory2', '')}
        SubCategory2_Prefix: {row.get('SubCategory2_Prefix ', '')}
        SubCategory2_Keywords: {row.get('SubCategory2_Keywords', '')}
        """.strip()
    
    def create_labels(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create label mappings for hierarchical classification"""
        primary_labels = df[self.config['classification']['primary_field']].unique().tolist()
        secondary_labels = df[self.config['classification']['secondary_field']].unique().tolist()
        
        # Create label encodings
        primary_label_map = {label: idx for idx, label in enumerate(primary_labels)}
        secondary_label_map = {label: idx for idx, label in enumerate(secondary_labels)}
        
        return {
            'primary_labels': primary_labels,
            'secondary_labels': secondary_labels,
            'primary_label_map': primary_label_map,
            'secondary_label_map': secondary_label_map
        }
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=df[self.config['classification']['primary_field']]
        )
        
        return train_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict:
        """Get class distribution for analysis"""
        primary_dist = df[self.config['classification']['primary_field']].value_counts()
        secondary_dist = df[self.config['classification']['secondary_field']].value_counts()
        
        return {
            'primary_distribution': primary_dist.to_dict(),
            'secondary_distribution': secondary_dist.to_dict(),
            'total_samples': len(df)
        }
