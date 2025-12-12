import re
import string
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_email(text):
    """
    Cleans email text by:
    - Converting to lowercase
    - Removing email addresses
    - Removing URLs
    - Removing punctuation
    - Tokenizing
    - Removing stopwords
    - Lemmatizing words
    """
    # Handle missing/null values
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove email addresses
    text = re.sub(r'\S+@\S+', "", text)
    
    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # 4. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 5. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 6. Tokenize
    words = text.split()
    
    # 7. Remove stopwords and lemmatize
    processed_words = []
    for w in words:
        if w not in stop_words and len(w) > 2:
            processed_words.append(lemmatizer.lemmatize(w))
    
    return " ".join(processed_words)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(r"D:\Infosys Springboard\Raw Dataset\customer_support_tickets_train.csv")  # Replace with your actual CSV filename

# Display basic information
print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values in key columns
print(f"\nMissing values in key columns:")
print(df[['subject', 'body', 'type', 'priority', 'language']].isnull().sum())

# Filter only English language emails
print("\n--- Filtering English Language Emails ---")
print(f"Total emails before filtering: {len(df)}")

# Filter for English (assuming 'en' or 'english' might be used)
df_english = df[df['language'].str.lower().str.strip() == 'en'].copy()

print(f"English emails found: {len(df_english)}")
print(f"Removed: {len(df) - len(df_english)} non-English emails")

if len(df_english) == 0:
    print("\nWARNING: No English emails found! Check the language column values:")
    print(df['language'].value_counts())
    exit()

# Select only required columns
df_processed = df_english[['subject', 'body', 'type', 'priority']].copy()

# Clean subject and body
print("\nCleaning subject and body texts...")
df_processed['cleaned_subject'] = df_processed['subject'].apply(clean_email)
df_processed['cleaned_body'] = df_processed['body'].apply(clean_email)

# Combine subject and body for a complete cleaned text
df_processed['cleaned_full_text'] = df_processed['cleaned_subject'] + ' ' + df_processed['cleaned_body']

# Remove leading/trailing spaces
df_processed['cleaned_full_text'] = df_processed['cleaned_full_text'].str.strip()

# Display type distribution
print("\n--- Type Distribution ---")
print(df_processed['type'].value_counts())
print("\nUnique types found:", df_processed['type'].nunique())

# Display priority distribution
print("\n--- Priority Distribution ---")
print(df_processed['priority'].value_counts())

# Create output directory for separate files
output_dir = "cleaned_by_type"
os.makedirs(output_dir, exist_ok=True)

# Save overall cleaned dataset
main_output = r"D:\Infosys Springboard\Cleaned Dataset\cleaned_email_dataset_all_types.csv"
df_processed.to_csv(main_output, index=False)
print(f"\n✓ Main cleaned dataset saved as '{main_output}'")

# Save separate files for each type
print(f"\n--- Creating Separate Files by Type ---")
type_counts = {}

for email_type in df_processed['type'].unique():
    # Skip if type is NaN
    if pd.isna(email_type):
        type_name = "unknown"
    else:
        # Clean the type name for filename (remove special characters)
        type_name = str(email_type).strip().replace(' ', '_').replace('/', '_')
    
    # Filter data for this type
    type_df = df_processed[df_processed['type'] == email_type].copy()
    type_counts[email_type] = len(type_df)
    
    # Save to separate file
    filename = f"{output_dir}/emails_type_{type_name}.csv"
    type_df.to_csv(filename, index=False)
    print(f"  ✓ Saved {len(type_df)} emails to '{filename}'")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total English emails processed: {len(df_processed)}")
print(f"Number of unique types: {df_processed['type'].nunique()}")
print(f"\nEmails by type:")
for email_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {email_type}: {count} emails")

print(f"\nAverage cleaned text length: {df_processed['cleaned_full_text'].str.split().str.len().mean():.2f} words")

# Display sample of final dataset
print("\n--- Sample of Processed Data ---")
print(df_processed[['subject', 'cleaned_subject', 'type', 'priority']].head(5))

# Create a summary report
summary_df = pd.DataFrame({
    'type': list(type_counts.keys()),
    'count': list(type_counts.values())
}).sort_values('count', ascending=False)

summary_df.to_csv(f"{output_dir}/type_summary.csv", index=False)
print(f"\n✓ Summary report saved as '{output_dir}/type_summary.csv'")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"Main file: {main_output}")
print(f"Type-specific files: {output_dir}/ directory")