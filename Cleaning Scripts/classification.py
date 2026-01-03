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


# Filter for English emails only
df_english = df[df['language'].str.lower().str.strip() == 'en'].copy()


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


# Create output directory for separate files
output_dir = "Cleaned by type"
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

