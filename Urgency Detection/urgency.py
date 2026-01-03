# ===============================
# Milestone 3: Urgency Detection
# ML + Rule-Based Hybrid Model
# ===============================

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report


# ===============================
# 1. LOAD DATASET
# ===============================
# CSV must contain:
# 1 column for text
# 1 column for urgency labels (low / medium / high)

DATA_PATH = r"D:\Infosys Springboard\Cleaned Dataset\cleaned_email_dataset_all_types.csv"        # <-- change path if needed
TEXT_COL = "cleaned_full_text"         # <-- change column name if needed
URGENCY_COL = "priority"         # <-- change column name if needed

df = pd.read_csv(DATA_PATH)


# ===============================
# 2. LABEL ENCODING
# ===============================
urgency_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

df["urgency_encoded"] = df[URGENCY_COL].map(urgency_map)

X_text = df[TEXT_COL]
y_urgency = df["urgency_encoded"]


# ===============================
# 3. TEXT CLEANING
# ===============================
def clean_email(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

X_cleaned = X_text.apply(clean_email)


# ===============================
# 4. TF-IDF VECTORIZATION
# ===============================
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X_urgency = vectorizer.fit_transform(X_cleaned)


# ===============================
# 5. TRAIN / TEST SPLIT
# ===============================
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_urgency,
    y_urgency,
    test_size=0.25,
    random_state=42,
    stratify=y_urgency
)


# ===============================
# 6. ML URGENCY MODEL
# ===============================
urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_train_u, y_train_u)


# ===============================
# 7. RULE-BASED URGENCY
# ===============================
high_urgency_keywords = [
    "urgent", "asap", "immediately", "not working", "down", "failed"
]

medium_urgency_keywords = [
    "soon", "please", "request", "help", "issue", "refund"
]

def rule_based_urgency(text):
    text = text.lower()

    for word in high_urgency_keywords:
        if word in text:
            return "high"

    for word in medium_urgency_keywords:
        if word in text:
            return "medium"

    return "low"


# ===============================
# 8. HYBRID URGENCY DETECTION
# ===============================
reverse_urgency_map = {0: "low", 1: "medium", 2: "high"}

def hybrid_urgency_detection(text):
    # Step 1: Rule-based
    rule_result = rule_based_urgency(text)
    if rule_result == "high":
        return "high"

    # Step 2: ML-based
    cleaned = clean_email(text)
    vec = vectorizer.transform([cleaned])
    ml_pred = urgency_model.predict(vec)[0]

    return reverse_urgency_map[ml_pred]


# ===============================
# 9. MODEL EVALUATION
# ===============================
y_pred_u = urgency_model.predict(X_test_u)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_u, y_pred_u))

f1 = f1_score(y_test_u, y_pred_u, average="weighted")
print(f"\nF1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test_u,
    y_pred_u,
    target_names=["low", "medium", "high"]
))


# ===============================
# 10. SAMPLE TESTING
# ===============================
print("\nHybrid Prediction Examples:")
print("1.", hybrid_urgency_detection("System is down since morning"))
print("2.", hybrid_urgency_detection("Please help with refund status"))
print("3.", hybrid_urgency_detection("Thanks for the update"))
