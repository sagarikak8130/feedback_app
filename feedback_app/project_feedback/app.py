from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load models with error handling
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Prepare few-shot urgency examples
urgency_examples: Dict[str, List[str]] = {
    'high': ["payment failed", "delivery issue", "urgent help needed"],
    'medium': ["app slow", "login improvement", "minor bug"],
    'low': ["add dark mode", "more payment options", "nice experience"]
}
urgency_embeddings = {k: embed_model.encode(v) for k, v in urgency_examples.items()}

# Define rule-based responses
response_templates: Dict[str, str] = {
    'complaint': "Thank you for contacting us. We're sorry to hear about your issue. Our team is reviewing your concern and will get back to you shortly.",
    'suggestion': "We appreciate your suggestion! Our team will review it for possible future improvements. Thank you for helping us get better.",
    'neutral': "Thank you for your feedback. We appreciate you taking the time to connect with us."
}

def classify_intent(text: str) -> str:
    """
    Classify the intent of the feedback text using zero-shot classification.
    
    Args:
        text (str): The feedback text to classify
        
    Returns:
        str: The predicted intent (complaint, suggestion, or neutral)
    """
    try:
        candidate_labels = ['complaint', 'suggestion', 'neutral']
        result = zero_shot_classifier(text, candidate_labels)
        return result['labels'][0]
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return 'neutral'  # Fallback to neutral in case of error

def detect_urgency(text: str) -> str:
    """
    Detect the urgency level of the feedback text using semantic similarity.
    
    Args:
        text (str): The feedback text to analyze
        
    Returns:
        str: The predicted urgency level (high, medium, or low)
    """
    try:
        text_emb = embed_model.encode([text])
        scores = {
            urgency: np.mean(cosine_similarity(text_emb, [emb for emb in emb_list]))
            for urgency, emb_list in urgency_embeddings.items()
        }
        return max(scores, key=scores.get)
    except Exception as e:
        logger.error(f"Error in urgency detection: {str(e)}")
        return 'medium'  # Fallback to medium urgency in case of error

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_feedback():
    if request.method == 'POST':
        try:
            feedback_text = request.form.get('feedback', '').strip()
            
            if not feedback_text:
                return render_template('home.html', error="Please provide feedback text")
            
            # Detect Intent and Urgency
            predicted_intent = classify_intent(feedback_text)
            predicted_urgency = detect_urgency(feedback_text)
            
            logger.info(f"Feedback analyzed - Intent: {predicted_intent}, Urgency: {predicted_urgency}")
            
            # Select rule-based response
            generated_response = response_templates[predicted_intent.lower()]
            
            return render_template('result.html',
                                generated_response=generated_response)
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return render_template('home.html', error="An error occurred while processing your feedback")

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)