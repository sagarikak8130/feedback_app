import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Settings
num_records = 100
channels = ['email', 'social_media', 'website_form']

# Sample feedback templates
complaints = [
    "My payment failed while checking out.",
    "The app crashes every time I open it.",
    "I have not received my order yet.",
    "Customer service is not responding to my emails.",
    "There was an error during the signup process.",
    "The delivery was delayed by more than a week.",
    "The product I received is defective.",
    "Refund process is taking too long."
]

suggestions = [
    "Please add more payment options.",
    "Introduce dark mode in the app.",
    "Add a live chat support option.",
    "Would love to see more color variants.",
    "Can you provide a loyalty rewards program?",
    "Please allow login using social accounts.",
    "It would be great to have offline access.",
    "Kindly expand your delivery areas."
]

neutral_feedback = [
    "Good experience overall, keep it up!",
    "Thanks for the quick support.",
    "Your website is very easy to navigate.",
    "Loved the packaging of the product.",
    "App update installed smoothly.",
    "No issues so far, satisfied with the service.",
    "App UI looks clean and user-friendly.",
    "Delivery agent was very polite."
]

# Combine all templates
all_feedback = complaints + suggestions + neutral_feedback

# Create records
records = []

for i in range(1, num_records + 1):
    message_type = random.choice(['complaint', 'suggestion', 'neutral'])
    
    if message_type == 'complaint':
        message = random.choice(complaints)
    elif message_type == 'suggestion':
        message = random.choice(suggestions)
    else:
        message = random.choice(neutral_feedback)
    
    record = {
        'id': i,
        'channel': random.choice(channels),
        'message': message,
        'timestamp': fake.date_time_between(start_date='-30d', end_date='now').strftime("%Y-%m-%d %H:%M:%S")
    }
    records.append(record)

# Save to CSV
df = pd.DataFrame(records)
df.to_csv('generated_customer_feedback.csv', index=False)

print("✅ Successfully generated 'generated_customer_feedback.csv' with", len(df), "records!")