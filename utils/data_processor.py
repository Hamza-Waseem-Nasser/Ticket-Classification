"""
Data Processor for Saber Ticket Classification
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import re

class DataProcessor:
    """Handles data loading and preprocessing for ticket classification"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_categories(self, file_path: str) -> pd.DataFrame:
        """Load Saber categories from CSV file"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            self.logger.info(f"Loaded {len(df)} categories from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load categories: {e}")
            return pd.DataFrame()
    
    def load_tickets(self, file_path: str, limit: int = None) -> pd.DataFrame:
        """Load cleaned tickets from CSV file"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if limit:
                df = df.head(limit)
            self.logger.info(f"Loaded {len(df)} tickets from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load tickets: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def preprocess_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess category data"""
        # Clean text columns
        text_columns = ['SubCategory', 'SubCategory2', 'SubCategory2_Keywords']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['SubCategory', 'SubCategory2'])
        
        return df
    
    def preprocess_tickets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess ticket data"""
        # Clean description column
        if 'Cleaned_Description' in df.columns:
            df['Cleaned_Description'] = df['Cleaned_Description'].apply(self.clean_text)
        
        # Remove empty descriptions
        df = df[df['Cleaned_Description'].str.len() > 0]
        
        return df
    
    def load_config(self, config_path: str = "config/config.yaml") -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
