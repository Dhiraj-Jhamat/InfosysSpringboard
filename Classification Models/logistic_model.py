from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


dataset = pd.read_csv(r'D:\Infosys Springboard\Cleaned Dataset\cleaned_email_dataset_all_types.csv')

df = pd.DataFrame(dataset)
x = df['cleaned_full_text']
y = df['type']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

# Fit on training data and transform
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

from sklearn. metrics import accuracy_score, classification_report
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))