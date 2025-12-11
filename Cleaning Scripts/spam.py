import re
import string
import pandas as pd
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
    
    # Convert to string (in case of numeric values)
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove email addresses
    text = re.sub(r'\S+@\S+', "", text)
    
    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # 4. Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    
    # 5. Remove numbers (optional - uncomment if needed)
    # text = re.sub(r'\d+', '', text)
    
    # 6. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 7. Tokenize
    words = text.split()
    
    # 8. Remove stopwords and lemmatize
    processed_words = []
    for w in words:
        if w not in stop_words and len(w) > 2:  # Also remove very short words
            processed_words.append(lemmatizer.lemmatize(w))
    
    return " ".join(processed_words)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(r"D:\Infosys Springboard\Raw Dataset\lingSpam.csv")  # Replace with your actual CSV filename

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Clean the Body column
print("\nCleaning email bodies...")
df['cleaned_text'] = df['Body'].apply(clean_email)

# Convert label to readable format (1=spam, 0=ham)
df['label_text'] = df['Label'].map({1: 'spam', 0: 'ham'})

# Display some examples
print("\n--- Sample Cleaned Emails ---")
for i in range(min(3, len(df))):
    print(f"\nOriginal: {df['Body'].iloc[i][:100]}...")
    print(f"Cleaned: {df['cleaned_text'].iloc[i]}")
    print(f"Label: {df['label_text'].iloc[i]}")

# Remove rows with empty cleaned text (if any)
initial_count = len(df)
df = df[df['cleaned_text'].str.strip() != '']
removed_count = initial_count - len(df)
if removed_count > 0:
    print(f"\nRemoved {removed_count} rows with empty cleaned text")

# Display label distribution
print("\n--- Label Distribution ---")
print(df['Label'].value_counts())
print("\nPercentage:")
print(df['Label'].value_counts(normalize=True) * 100)

# Create final structured dataset
final_df = pd.DataFrame({
    'orginal_text': df['Body'],
    'cleaned_text': df['cleaned_text'],
    'label': df['Label'],
    'label_text': df['label_text']
})

# Save the cleaned dataset
output_filename = r"D:\Infosys Springboard\Cleaned Dataset\cleaned_spam.csv"
final_df.to_csv(output_filename, index=False)
print(f"\n✓ Cleaned dataset saved as '{output_filename}'")

# Display final statistics
print("\n--- Final Dataset Statistics ---")
print(f"Total emails: {len(final_df)}")
print(f"Spam emails: {(final_df['label'] == 1).sum()}")
print(f"Ham emails: {(final_df['label'] == 0).sum()}")
print(f"\nAverage cleaned text length: {final_df['cleaned_text'].str.split().str.len().mean():.2f} words")

# Display sample of final dataset
print("\n--- Final Dataset Preview ---")
print(final_df.head(10))