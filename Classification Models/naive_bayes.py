# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris

# # 1. Load sample data (Iris dataset)
# iris = load_iris()
# X = iris.data
# y = iris.target
# print(f"Dataset features shape: {X.shape}, labels shape: {y.shape}")

# # 2. Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(f"Training set size: {len(X_train)} samples")
# print(f"Testing set size: {len(X_test)} samples")

# # 3. Initialize and train the Gaussian Naive Bayes model
# model = GaussianNB()
# model.fit(X_train, y_train)

# # 4. Make predictions on the test set
# y_pred = model.predict(X_test)

# # 5. Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# # 6. Predict on a single new instance
# new_data_point = [[5.1, 3.5, 1.4, 0.2]] # Example data for an Iris flower
# prediction = model.predict(new_data_point)
# print(f"Prediction for the new data point: {iris.target_names[prediction[0]]}")
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

