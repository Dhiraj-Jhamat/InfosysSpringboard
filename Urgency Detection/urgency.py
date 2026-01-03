# Milestone 3 (Weeks 5–6): Urgency Detection & Scoring
# ●	Objective: Implement urgency prediction.
# ●	Tasks:
# ○	Train urgency classification model.
# ○	Combine ML + keyword-based detection.
# Validate results with confusion matrix & F1 score.

# urgency labels aligned with emails
urgency_labels = [
    "high",     # internet not working immediately
    "medium",   # refund request
    "low",      # appreciation
    "low"       # spam / unsubscribe
]

urgency_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

y_urgency = [urgency_map[u] for u in urgency_labels]

#step2: rule based urgency detection

high_urgency_keywords = [
    "urgent", "asap", "immediately", "not working", "down", "failed"
]

medium_urgency_keywords = [
    "soon", "please", "request", "help", "issue"
]

# rule-based urgency detection function

def rule_based_urgency(text):
    text = text.lower()
    
    for word in high_urgency_keywords:
        if word in text:
            return "high"

    for word in medium_urgency_keywords:
        if word in text:
            return "medium"
    
    return "low"

print(rule_based_urgency("Internet not working please fix asap"))
print(rule_based_urgency("Need help with refund"))
print(rule_based_urgency("Thank you for the support"))

## high, medium, low

# combine ML + rule-based approach

## reuse the TF-IDF features because- same text -> diff tasks -> diff labels, features stays the same, the targets change

X_urgency= vectorizer.fit_transform(cleaned_emails)

# train/test spilt

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_urgency, y_urgency, test_size=0.25, random_state=42)


# train logistic regression for urgency

urgency_model =LogisticRegression(max_iter= 1000)
urgency_model.fit(X_train_u, y_train_u)

# pred urgency 

y_pred_u= urgency_model.predict(X_test_u)


## hybrid - urgency scoring (rule + ML)

reverse_urgency_map = {0: "low", 1: "medium", 2: "high"}

def hybrid_urgency_detection(text):
    # step 1: rule-based check 
    
    rule_result= rule_based_urgency(text)
    
    if rule_result == "high":
        return "high"
    
    # step2: ML prediction
    
    cleaned= clean_email(text)
    vec= vectorizer.transform ([cleaned])
    ml_pred = urgency_model.predict(vec)[0]
    
    return reverse_urgency_map[ml_pred]


print(hybrid_urgency_detection("System is down since morning"))
print(hybrid_urgency_detection("Please help with refund status"))
print(hybrid_urgency_detection("Thanks for the update"))


# step 5: evaluation using confusion matrix + F1 score 

from sklearn.metrics import confusion_matrix, f1_score

print(confusion_matrix(y_test_u, y_pred_u))
f1= f1_score(y_test_u, y_pred_u, average="weighted")
print(f"F1 Score: {f1}")