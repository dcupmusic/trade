from flask import Flask, request, jsonify
import joblib
import torch.nn as nn

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.joblib')

# Define the API endpoint for making predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract features from URL parameters
        timestamp = float(request.args.get('timestamp'))
        price = float(request.args.get('price'))
        rsi = float(request.args.get('rsi'))

        # Make a prediction using the loaded model
        prediction = model.predict([[timestamp, price, rsi]])
        
        # Convert the prediction to a string (assuming it's binary)
        prediction_str = str(prediction[0])

        return jsonify({'prediction': prediction_str})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
