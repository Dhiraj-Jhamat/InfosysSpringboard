import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialising stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_email(text):

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
        
    # 5. Remove punctuation
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
df = pd.read_csv(r"D:\Infosys Springboard\Raw Dataset\spam_ham_dataset.csv")  

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Clean the Body column
print("\nCleaning email bodies...")
df['cleaned_text'] = df['text'].apply(clean_email)

# Create final structured dataset
final_df = pd.DataFrame({
    'orginal_text': df['text'],
    'cleaned_text': df['cleaned_text'],
    'label': df['label'],
})

# Save the cleaned dataset
output_filename = r"D:\Infosys Springboard\Cleaned Dataset\cleaned_spam.csv"
final_df.to_csv(output_filename, index=False)
print(f"\n✓ Cleaned dataset saved as '{output_filename}'")

# Display sample of final dataset
print("\n--- Final Dataset Preview ---")
print(final_df.head(10))