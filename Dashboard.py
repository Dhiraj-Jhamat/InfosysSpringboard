import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================================
# MODEL PATH CONFIGURATION - ADD YOUR MODEL PATHS HERE
# ============================================================================

# Classification Models
NAIVE_BAYES_MODEL_PATH = "Classification Models/naive_bayes_model.pkl"
LOGISTIC_REGRESSION_MODEL_PATH = r"Classification Models/logisticmodel.pkl"
BERT_MODEL_PATH = r"final_bert_model"  # Directory containing BERT model files

# Vectorizer for Naive Bayes and Logistic Regression (if using TF-IDF/CountVectorizer)
VECTORIZER_PATH = r"Classification Models/tfidf_vectorizer.pkl"

# Rule-based Urgency Detection Model
URGENCY_MODEL_PATH = r"Urgency Detection/urgency_pipeline.pkl"

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models and return them as a dictionary"""
    models = {}
    
    try:
        # Load Naive Bayes Model
        with open(NAIVE_BAYES_MODEL_PATH, 'rb') as f:
            models['naive_bayes'] = pickle.load(f)
    except Exception as e:
        models['naive_bayes'] = None
        st.warning(f"Failed to load Naive Bayes model: {e}")
    
    try:
        # Load Logistic Regression Model
        with open(LOGISTIC_REGRESSION_MODEL_PATH, 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
    except Exception as e:
        models['logistic_regression'] = None
        st.warning(f"Failed to load Logistic Regression model: {e}")
    
    try:
        # Load BERT Model
        models['bert_tokenizer'] = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        models['bert_model'] = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        models['bert_model'].eval()
    except Exception as e:
        models['bert_tokenizer'] = None
        models['bert_model'] = None
        st.warning(f"Failed to load BERT model: {e}")
    
    try:
        # Load Vectorizer (for Naive Bayes and Logistic Regression)
        with open(VECTORIZER_PATH, 'rb') as f:
            models['vectorizer'] = pickle.load(f)
    except Exception as e:
        models['vectorizer'] = None
        st.warning(f"Failed to load Vectorizer: {e}")
    
    try:
        # Load Urgency Detection Model
        with open(URGENCY_MODEL_PATH, 'rb') as f:
            models['urgency_model'] = pickle.load(f)
    except Exception as e:
        models['urgency_model'] = None
        st.warning(f"Failed to load Urgency model: {e}")
    
    return models

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_naive_bayes(text, models):
    """Predict using Naive Bayes model"""
    if models['naive_bayes'] is None or models['vectorizer'] is None:
        return "Model not loaded"
    
    try:
        text_vectorized = models['vectorizer'].transform([text])
        prediction = models['naive_bayes'].predict(text_vectorized)[0]
        proba = models['naive_bayes'].predict_proba(text_vectorized)[0]
        confidence = max(proba) * 100
        return prediction, confidence
    except Exception as e:
        return f"Error: {e}", 0

def predict_logistic_regression(text, models):
    """Predict using Logistic Regression model"""
    if models['logistic_regression'] is None or models['vectorizer'] is None:
        return "Model not loaded"
    
    try:
        text_vectorized = models['vectorizer'].transform([text])
        prediction = models['logistic_regression'].predict(text_vectorized)[0]
        proba = models['logistic_regression'].predict_proba(text_vectorized)[0]
        confidence = max(proba) * 100
        return prediction, confidence
    except Exception as e:
        return f"Error: {e}", 0

def predict_bert(text, models):
    """Predict using BERT model"""
    if models['bert_model'] is None or models['bert_tokenizer'] is None:
        return "Model not loaded"
    
    try:
        # Tokenize input
        inputs = models['bert_tokenizer'](
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = models['bert_model'](**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        return prediction, confidence
    except Exception as e:
        return f"Error: {e}", 0

def predict_urgency(text, models):
    """Predict urgency using rule-based model"""
    if models['urgency_model'] is None:
        return "Model not loaded"
    
    try:
        # If it's a rule-based model, it might have a predict method
        # Adjust this based on your actual urgency model implementation
        urgency = models['urgency_model'].predict([text])[0]
        return urgency
    except Exception as e:
        return f"Error: {e}"

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Email Classifier & Urgency Detector",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Email Classification & Urgency Detection")
    st.markdown("---")
    
    # Load models
    models = load_models()
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Email Input")
        subject = st.text_input("Email Subject:", placeholder="Enter email subject...")
        content = st.text_area("Email Content:", height=200, placeholder="Enter email content...")
    
    with col2:
        st.subheader("Instructions")
        st.info("""
        **How to use:**
        1. Enter the email subject
        2. Enter the email content
        3. Click 'Classify Email' to see predictions
        
        The system will:
        - Classify the email using 3 different models
        - Detect the urgency level
        """)
    
    st.markdown("---")
    
    # Classify button
    if st.button("🔍 Classify Email", type="primary", use_container_width=True):
        if not subject.strip() and not content.strip():
            st.error("⚠️ Please enter at least subject or content!")
        else:
            # Concatenate subject and content
            combined_text = f"{subject} {content}".strip()
            
            with st.spinner("Analyzing email..."):
                # Create columns for results
                st.subheader("Classification Results")
                col1, col2, col3 = st.columns(3)
                
                # Naive Bayes Prediction
                with col1:
                    st.markdown("### 🔹 Naive Bayes")
                    nb_result = predict_naive_bayes(combined_text, models)
                    if isinstance(nb_result, tuple):
                        st.metric("Prediction", nb_result[0])
                        st.metric("Confidence", f"{nb_result[1]:.2f}%")
                    else:
                        st.error(nb_result)
                
                # Logistic Regression Prediction
                with col2:
                    st.markdown("### 🔹 Logistic Regression")
                    lr_result = predict_logistic_regression(combined_text, models)
                    if isinstance(lr_result, tuple):
                        st.metric("Prediction", lr_result[0])
                        st.metric("Confidence", f"{lr_result[1]:.2f}%")
                    else:
                        st.error(lr_result)
                
                # BERT Prediction
                with col3:
                    st.markdown("### 🔹 BERT")
                    bert_result = predict_bert(combined_text, models)
                    if isinstance(bert_result, tuple):
                        st.metric("Prediction", bert_result[0])
                        st.metric("Confidence", f"{bert_result[1]:.2f}%")
                    else:
                        st.error(bert_result)
                
                st.markdown("---")
                
                # Urgency Detection
                st.subheader("Urgency Detection")
                urgency_result = predict_urgency(combined_text, models)
                
                if "Error" not in str(urgency_result) and "not loaded" not in str(urgency_result):
                    # Display urgency with color coding
                    if "high" in str(urgency_result).lower() or "urgent" in str(urgency_result).lower():
                        st.error(f"🔴 Urgency Level: {urgency_result}")
                    elif "medium" in str(urgency_result).lower() or "moderate" in str(urgency_result).lower():
                        st.warning(f"🟡 Urgency Level: {urgency_result}")
                    else:
                        st.success(f"🟢 Urgency Level: {urgency_result}")
                else:
                    st.error(urgency_result)
                
                # Show the combined text used for prediction
                with st.expander("📝 View Combined Text"):
                    st.text_area("Text used for classification:", combined_text, height=150, disabled=True)

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This dashboard uses multiple machine learning models to classify emails:
        
        **Classification Models:**
        - Naive Bayes
        - Logistic Regression
        - BERT (Transformer-based)
        
        **Urgency Detection:**
        - Rule-based model
        
        All models work on the concatenated text of email subject and content.
        """)
        
        st.markdown("---")
        st.markdown("### Model Status")
        
        # Show model loading status
        if models['naive_bayes'] is not None:
            st.success("✅ Naive Bayes loaded")
        else:
            st.error("❌ Naive Bayes not loaded")
        
        if models['logistic_regression'] is not None:
            st.success("✅ Logistic Regression loaded")
        else:
            st.error("❌ Logistic Regression not loaded")
        
        if models['bert_model'] is not None:
            st.success("✅ BERT loaded")
        else:
            st.error("❌ BERT not loaded")
        
        if models['urgency_model'] is not None:
            st.success("✅ Urgency Model loaded")
        else:
            st.error("❌ Urgency Model not loaded")

if __name__ == "__main__":
    main()
