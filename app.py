from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Ensure the app variable is named exactly 'app' for Gunicorn
app = Flask(__name__)

# Load the Gaussian Naive Bayes model
try:
    with open('naive_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'naive_model.pkl' not found.")

# The UI Layout built directly into the app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Purchase Prediction Model</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
        h2 { text-align: center; color: #1a1a1a; margin-top: 0; }
        .form-group { margin-bottom: 20px; }
        label { font-weight: bold; display: block; margin-bottom: 8px; color: #444; }
        input, select { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; font-size: 16px; }
        button { width: 100%; padding: 14px; background-color: #0d6efd; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: 0.3s; }
        button:hover { background-color: #0b5ed7; }
        .result-box { margin-top: 25px; padding: 15px; text-align: center; border-radius: 6px; font-size: 18px; font-weight: bold; }
        .purchased { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
        .not-purchased { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Purchase Predictor</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="Gender">Gender</label>
                <select name="Gender" id="Gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="Age">Age</label>
                <input type="number" id="Age" name="Age" required placeholder="e.g. 28" min="18" max="100">
            </div>
            
            <div class="form-group">
                <label for="EstimatedSalary">Estimated Salary</label>
                <input type="number" id="EstimatedSalary" name="EstimatedSalary" required placeholder="e.g. 50000" min="0">
            </div>
            
            <button type="submit">Predict Purchase</button>
        </form>

        {% if prediction_text %}
            <div class="result-box {% if prediction_value == 1 %}purchased{% else %}not-purchased{% endif %}">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    # Renders the UI when you visit the base URL
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the 3 specific features required by the model
        gender = float(request.form.get('Gender'))
        age = float(request.form.get('Age'))
        salary = float(request.form.get('EstimatedSalary'))
        
        # Format for GaussianNB
        features = np.array([[gender, age, salary]])
        
        # Predict
        prediction = model.predict(features)
        output = int(prediction[0])
        
        # Determine the display text based on the 0 or 1 output
        if output == 1:
            result_text = "Result: User WILL Purchase"
        else:
            result_text = "Result: User WILL NOT Purchase"
            
        return render_template_string(HTML_TEMPLATE, prediction_text=result_text, prediction_value=output)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
