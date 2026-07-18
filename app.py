from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Ensure the instance name is exactly 'app' for Gunicorn
app = Flask(__name__)

# Load your Naive Bayes model (ensure naive_model.pkl is in your root directory)
try:
    with open('naive_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'naive_model.pkl' not found. Please ensure the file is in the correct directory.")

# 1. Root Route: This renders your front-end UI
@app.route('/')
def home():
    # Render looks for index.html inside a folder named 'templates'
    return render_template('index.html')

# 2. Prediction Route: Processes the UI form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form fields (or request.json if sending JSON via JS)
        # Note: Handle Gender encoding depending on how your model was trained (e.g., Male=1, Female=0)
        gender = float(request.form.get('Gender', 0))
        age = float(request.form.get('Age', 0))
        salary = float(request.form.get('EstimatedSalary', 0))
        
        # Format the features into a 2D array for the scikit-learn model
        features = np.array([[gender, age, salary]])
        
        # Make the prediction
        prediction = model.predict(features)
        output = int(prediction[0])
        
        # Return the result back to your UI
        # You can either render a template with the result or return JSON
        return render_template('index.html', prediction_text=f'Purchased Status Prediction: {output}')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Standard local running configuration
    app.run(host='0.0.0.0', port=5000, debug=True)
