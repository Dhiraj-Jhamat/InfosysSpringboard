# from datasets import load_dataset

# ds = load_dataset("Tobi-Bueck/customer-support-tickets")

# # print(ds["train"])

# train_ds = ds["train"]

# train_ds.to_csv("customer_support_tickets_train.csv", index=False)

import pandas as pd
import re
import string
from typing import Optional

class EmailDataPreprocessor:
    """
    A class to handle reading, preprocessing, and merging email datasets
    """
    
    def __init__(self):
        self.stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        ])
    
    def remove_email_signatures(self, text: str) -> str:
        """
        Remove common email signatures and footers
        """
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove common signature patterns
        signature_patterns = [
            r'--+\s*\n.*',  # Lines starting with --
            r'Best regards.*',
            r'Sincerely.*',
            r'Thanks.*\n.*',
            r'Sent from my.*',
            r'Get Outlook for.*',
            r'Kind regards.*',
            r'Warm regards.*',
            r'Cheers.*'
        ]
        
        for pattern in signature_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text
    
    def remove_email_addresses(self, text: str) -> str:
        """
        Remove email addresses from text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text
    
    def normalize_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Normalize text: lowercase, remove punctuation, extra spaces
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Optionally remove stopwords
        if remove_stopwords:
            words = text.split()
            text = ' '.join([word for word in words if word not in self.stopwords])
        
        return text
    
    def preprocess_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Complete preprocessing pipeline for a single text
        """
        text = self.remove_html_tags(text)
        text = self.remove_email_signatures(text)
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        text = self.normalize_text(text, remove_stopwords)
        return text
    
    def read_dataset1(self, filepath: str, filter_english: bool = True) -> pd.DataFrame:
        """
        Read the first dataset with columns:
        subject, body, answer, type, queue, priority, language, version, tag_1-8
        
        Args:
            filepath: Path to the CSV file
            filter_english: If True, only keep rows where language is 'english'
        
        Returns:
            DataFrame with selected columns
        """
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Filter for English only if specified
        if filter_english and 'language' in df.columns:
            df = df[df['language'].str.lower() == 'english'].copy()
        
        # Select required columns
        columns_to_keep = ['subject', 'body', 'type', 'priority']
        
        # Keep only columns that exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df_selected = df[existing_columns].copy()
        
        # Add source identifier
        df_selected['source'] = 'dataset1'
        
        return df_selected
    
    def read_dataset2(self, filepath: str) -> pd.DataFrame:
        """
        Read the second dataset with columns:
        (index), Body, Label
        
        Args:
            filepath: Path to the CSV file
        
        Returns:
            DataFrame with mapped columns
        """
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Create dataframe with required structure
        df_mapped = pd.DataFrame({
            'subject': None,  # No subject in this dataset
            'body': df['Body'],
            'type': df['Label'],  # Map Label to type
            'priority': None,  # No priority in this dataset
            'source': 'dataset2'
        })
        
        return df_mapped
    
    def merge_datasets(self, dataset1_path: str, dataset2_path: str, 
                      output_path: str, preprocess: bool = True) -> pd.DataFrame:
        """
        Read both datasets, merge them, and save to a single file
        
        Args:
            dataset1_path: Path to first CSV file
            dataset2_path: Path to second CSV file
            output_path: Path to save merged CSV file
            preprocess: If True, apply preprocessing to text columns
        
        Returns:
            Merged DataFrame
        """
        print("Reading Dataset 1...")
        df1 = self.read_dataset1(dataset1_path, filter_english=True)
        print(f"Dataset 1: {len(df1)} rows loaded (English only)")
        
        print("\nReading Dataset 2...")
        df2 = self.read_dataset2(dataset2_path)
        print(f"Dataset 2: {len(df2)} rows loaded")
        
        # Merge datasets
        print("\nMerging datasets...")
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Apply preprocessing if requested
        if preprocess:
            print("\nPreprocessing text data...")
            print("  - Cleaning subject column...")
            merged_df['subject_clean'] = merged_df['subject'].apply(
                lambda x: self.preprocess_text(x, remove_stopwords=False)
            )
            
            print("  - Cleaning body column...")
            merged_df['body_clean'] = merged_df['body'].apply(
                lambda x: self.preprocess_text(x, remove_stopwords=False)
            )
        
        # Reorder columns
        column_order = ['subject', 'body', 'type', 'priority', 'source']
        if preprocess:
            column_order.extend(['subject_clean', 'body_clean'])
        
        merged_df = merged_df[column_order]
        
        # Save to file
        print(f"\nSaving merged dataset to {output_path}...")
        merged_df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MERGE SUMMARY")
        print("="*60)
        print(f"Total rows: {len(merged_df)}")
        print(f"From Dataset 1: {len(df1)}")
        print(f"From Dataset 2: {len(df2)}")
        print(f"\nNull values per column:")
        print(merged_df.isnull().sum())
        print(f"\nUnique types: {merged_df['type'].nunique()}")
        print(f"Type distribution:\n{merged_df['type'].value_counts()}")
        if 'priority' in merged_df.columns:
            print(f"\nUnique priorities: {merged_df['priority'].nunique()}")
            print(f"Priority distribution:\n{merged_df['priority'].value_counts()}")
        print("="*60)
        
        return merged_df


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = EmailDataPreprocessor()
    
    # Define file paths
    dataset1_path = "D:\Infosys Springboard\Raw Dataset\customer_support_tickets_train.csv"  # Your first dataset
    dataset2_path = "D:\Infosys Springboard\Raw Dataset\lingSpam.csv"  # Your second dataset
    output_path = "merged_email_dataset.csv"
    
    # Merge datasets with preprocessing
    merged_data = preprocessor.merge_datasets(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        output_path=output_path,
        preprocess=True
    )
    
    # Display first few rows
    print("\nFirst 5 rows of merged dataset:")
    print(merged_data.head())
    
    # Optional: Save preprocessed version for ML training
    # Keep only clean columns for modeling
    ml_ready_data = merged_data[['subject_clean', 'body_clean', 'type', 'priority']].copy()
    ml_ready_data.to_csv("ml_ready_dataset.csv", index=False)
    print("\nML-ready dataset saved to ml_ready_dataset.csv")