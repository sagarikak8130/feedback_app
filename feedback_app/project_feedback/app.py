from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load models once (Important: don't reload inside each request!)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Prepare few-shot urgency examples
urgency_examples = {
    'high': ["payment failed", "delivery issue", "urgent help needed"],
    'medium': ["app slow", "login improvement", "minor bug"],
    'low': ["add dark mode", "more payment options", "nice experience"]
}
urgency_embeddings = {k: embed_model.encode(v) for k, v in urgency_examples.items()}

# Define rule-based responses
response_templates = {
    'complaint': "Thank you for contacting us. We're sorry to hear about your issue. Our team is reviewing your concern and will get back to you shortly.",
    'suggestion': "We appreciate your suggestion! Our team will review it for possible future improvements. Thank you for helping us get better.",
    'neutral': "Thank you for your feedback. We appreciate you taking the time to connect with us."
}

def classify_intent(text):
    candidate_labels = ['complaint', 'suggestion', 'neutral']
    result = zero_shot_classifier(text, candidate_labels)
    return result['labels'][0]

def detect_urgency(text):
    text_emb = embed_model.encode([text])
    scores = {
        urgency: np.mean(cosine_similarity(text_emb, [emb for emb in emb_list]))
        for urgency, emb_list in urgency_embeddings.items()
    }
    return max(scores, key=scores.get)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_feedback():
    if request.method == 'POST':
        feedback_text = request.form['feedback']

        # Detect Intent
        predicted_intent = classify_intent(feedback_text)

        # Detect Urgency
        # text_emb = embed_model.encode([feedback_text])
        # scores = {
        #     urgency: np.mean(cosine_similarity(text_emb, [emb for emb in emb_list]))
        #     for urgency, emb_list in urgency_embeddings.items()
        # }
        predicted_urgency = detect_urgency(feedback_text)
        # predicted_urgency = max(scores, key=scores.get)
        
        print(predicted_intent,predicted_urgency)

        # Select rule-based response
        generated_response = response_templates[predicted_intent.lower()]

        return render_template('result.html',
                            #    message=feedback_text,
                            #    intent=predicted_intent,
                            #    urgency=predicted_urgency,
                               generated_response=generated_response)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)