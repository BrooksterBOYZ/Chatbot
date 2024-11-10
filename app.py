from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for enabling cross-origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()  # Avoid issues if 'message' is missing or empty
    
    # Check if input is empty
    if not user_input:
        return jsonify({"response": "Please provide a message!"})  # Respond if the message is empty
    
    print(f"Received user input: {user_input}")  # Debugging log

    # If user says "bye" or "goodbye", exit immediately
    if user_input.lower() in ["bye", "goodbye"]:
        print("User said goodbye")  # Debugging log
        return jsonify({"response": "Goodbye! It was nice talking to you."})
    
    # Clean and process the user input
    processed_input = process_input(user_input)
    print(f"Processed input: {processed_input}")  # Debugging log
    
    # Get the predicted intent for the input
    try:
        intent = get_intent(processed_input)
        print(f"Predicted intent: {intent}")  # Debugging log
    except Exception as e:
        print(f"Error predicting intent: {e}")  # Debugging log
        return jsonify({"response": "There was an error processing your request."})
    
    # Get a response based on the predicted intent
    response = get_response(intent)
    print(f"Response: {response}")  # Debugging log
    
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

