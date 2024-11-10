import spacy
import random
import json
import nltk
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Download nltk data
nltk.download('punkt')

# Load spaCy model for NLP tasks (e.g., tokenization, lemmatization)
nlp = spacy.load("en_core_web_sm")

# Define some intents (simplified example)
intents = {
    "greeting": ["hello", "hi", "hey", "howdy", "yo", "what's up"],
    "goodbye": ["bye", "goodbye", "see you later", "take care"],
    "thanks": ["thank you", "thanks", "appreciate it"],
    "weather": ["what's the weather like", "weather", "is it raining", "will it rain tomorrow?"],
    "name": ["what is your name", "who are you", "tell me your name"],
    "about": ["tell me about yourself", "what do you do?", "who created you"],
    "unknown": []
}

# Prepare training data
training_sentences = []
training_labels = []

for label, patterns in intents.items():
    for pattern in patterns:
        training_sentences.append(pattern)
        training_labels.append(label)

# Step 1: Vectorize sentences using CountVectorizer (bag of words)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Step 2: Train the classifier
X_train, X_test, y_train, y_test = train_test_split(training_sentences, training_labels, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Define responses based on detected intent
responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! How can I assist you?", "Hey! What can I do for you today?"],
    "goodbye": ["Goodbye! Have a great day!", "See you later! Take care!", "Bye! It was nice talking to you."],
    "thanks": ["You're welcome!", "Happy to help!", "No problem!"],
    "weather": ["I don't have live weather updates yet, but I hope it's sunny!", "You can check the weather using a weather API.", "I hope it's nice out there today!"],
    "name": ["I'm your friendly chatbot!", "I don't have a name, but you can call me Bot.", "Call me ChatBot!"],
    "about": ["I'm just a bot created to chat with you. How can I help?", "I'm here to answer your questions. What can I do for you?", "I was created to assist and chat with people like you!"],
    "unknown": ["Sorry, I didn't quite understand that. Can you rephrase?", "I'm not sure what you're asking. Could you clarify?", "I didn't get that. Can you try asking something else?"]
}

# Function to get the intent of a user message
def get_intent(user_input):
    # Predict intent using the model
    prediction = model.predict([user_input])
    return prediction[0]

# Function to get a response based on the detected intent
def get_response(intent):
    return random.choice(responses.get(intent, responses["unknown"]))

# Function to clean and process the user input using spaCy
def process_input(user_input):
    # Tokenize and lemmatize
    doc = nlp(user_input)
    processed_input = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return processed_input

# Chatbot loop
def start_chat():
    print("Chatbot: Hi! I'm a more advanced chatbot. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ").lower()
        
        if user_input in ["bye", "goodbye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break
        
        # Clean and process the user input
        processed_input = process_input(user_input)
        
        # Get the predicted intent for the input
        intent = get_intent(processed_input)
        
        # Get a response based on the predicted intent
        response = get_response(intent)
        
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    start_chat()
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Your existing code...

if __name__ == "__main__":
    app.run(debug=True)
