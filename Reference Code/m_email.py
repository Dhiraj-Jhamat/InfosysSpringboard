import re # regular expression - used to remove certain patterns (HTML, signatures, etc from the data)
import string # used to get the punctuation characters/ lists
from nltk.corpus import stopwords # remove unwanted common words 
from nltk.stem import WordNetLemmatizer #convert the words to the base form 

stop_words= set (stopwords.words("english")) # lists of common words like "the", "is" , "and"
lemmatizer= WordNetLemmatizer() # used to simplify the words(fixing-> fix)

# 1. cleaning the email text

def clean_email(text):
    # 1. Lowercase 
    text= text.lower() # lowers the text so that it doesn't give any confusion 
    #2. remove the email IDs
    text= re.sub(r'\S@\S+', "", text) #removes the irrelvant and nosy data 
    #3. remove the URLS
    text = re.sub(r"http\S+|www\S+", "", text) #removes the irrelvant and noisy url that adds no meaning to the classsification
    #4. remove the punctation 
    text= text.translate(str.maketrans("", "", string.punctuation))# reduces the noise like symbols- !, ? 
    #5. Tokenize
    words= text.split() #i can able to split tthe text into the "words"
    #6. remove the stopwords + lemmatization 
    processed_words= []
    for w in words: 
        if w not in stop_words:
            processed_words.append(lemmatizer.lemmatize(w)) 
            # purpose- removes the useless words, lemmatizes each words- running= run, issues= issue
    return " ".join(processed_words) # returns the cleaned sentence back as a text 
    
sample_email= "Hello team, My internet is not working. please fix this ASAP!!!!"
print(clean_email(sample_email))

#o/p: hello team internet work please fix asap 


# 2. A sample dataset of emails
# they must understand that email classification depends on the data + labels
# Helps them to see how the ML model learns from the data + labels
# required for ML model training

emails = [
    "My internet is not working, I need help immediately",
    "I want to know the status of my refund request",
    "Great service! I appreciate the quick support",
    "You guys keep sending too many mails. Stop it."
]

labels = [
    "complaint",
    "request",
    "feedback",
    "spam"
]

# cleaning the emails using the clean_email function

cleaned_emails= [clean_email(e) for e in emails]
print(cleaned_emails)

#o/p : 
# [
#  'internet work need help immediately',
#  'want know status refund request',
#  'great service appreciate quick support',
#  'guys keep sending many mail stop'
# ]

# Convert the text into numerical vectors using TF-IDF vectorization
# important step and cruical 
# ML models cannot understand the text directly, so that we are converting them to numbers 

## TF-IDF vectorization - learn 

from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer= TfidfVectorizer()

x = vectorizer.fit_transform(cleaned_emails)

# how the text becomes a numeric matrix 
# why TF-IDF is necessary- other ? alterative? 
# what is feature extractions means 

# 4. train/test spilt 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.25, random_state=42
)

# 5. Subject + Body extraction from email

subjects = [
    "Internet Issue",
    "Refund Request",
    "Appreciation",
    "Unwanted Emails"
]

full_emails = [subjects[i] + " " + emails[i] for i in range(len(emails))]
cleaned_emails = [clean_email(e) for e in full_emails]

# 6. Saving the structured cleaned data

import pandas as pd

df = pd.DataFrame({
    "subject": subjects,
    "body": emails,
    "cleaned_text": cleaned_emails,
    "label": labels
})

print(df)
df.to_csv("cleaned_email_dataset.csv", index=False)

# Optional: Add Urgency Labels

urgency = ["high", "medium", "low", "low"]
df["urgency"] = urgency


