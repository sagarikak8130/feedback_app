import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

# Step 1: Load your generated dataset
df = pd.read_csv('generated_customer_feedback.csv')  # Must have a 'message' column
print(f"Loaded {len(df)} records.")

# Step 2: Initialize models
print("Loading models...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Step 3: Embed all messages
print("Embedding messages...")
df['embedding'] = df['message'].apply(lambda x: embed_model.encode(x))

# Step 4: Zero-shot intent classification
print("Classifying intent...")
def classify_intent(text):
    candidate_labels = ['complaint', 'suggestion', 'neutral']
    result = zero_shot_classifier(text, candidate_labels)
    return result['labels'][0]  # Top prediction

df['intent'] = df['message'].apply(classify_intent)

# Step 5: Few-shot urgency classification
print("Detecting urgency levels...")
urgency_examples = {
    'high': ["payment failed", "delivery issue", "urgent help needed"],
    'medium': ["app slow", "login improvement", "minor bug"],
    'low': ["add dark mode", "more payment options", "nice experience"]
}
urgency_embeddings = {k: embed_model.encode(v) for k, v in urgency_examples.items()}

def detect_urgency(text):
    text_emb = embed_model.encode([text])
    scores = {
        urgency: np.mean(cosine_similarity(text_emb, [emb for emb in emb_list]))
        for urgency, emb_list in urgency_embeddings.items()
    }
    return max(scores, key=scores.get)

df['urgency'] = df['message'].apply(detect_urgency)

# Step 6: Optional - Clustering similar feedback
print("Running HDBSCAN clustering...")
X = np.vstack(df['embedding'].values)
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df['cluster'] = clusterer.fit_predict(X)

# Step 7: Save processed output
print("Saving final analyzed file...")
df.drop(columns=['embedding'], inplace=True)  # Optional: Remove embeddings column
df.to_csv('analyzed_customer_feedback.csv', index=False)

print("Output saved to analyzed_customer_feedback.csv")


# saving the hdbscan clustering method
import pickle

with open('hdbscan_clusterer.pkl', 'wb') as f:
    pickle.dump(clusterer, f)

print("HDBSCAN clusterer saved successfully.")
