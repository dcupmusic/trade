from flask import Flask, request, jsonify
import torch
import joblib
from lstm_v1_binary_bigger import BiLSTMClassifierWithAttention

app = Flask(__name__)

model = BiLSTMClassifierWithAttention(2, 32, 2, 1, 0.05)
model.load_state_dict(torch.load('trained_model_lstm_binary.pth'))
model.eval()

# Load the saved scaler parameters
scaler_params = joblib.load('scaler_params_binary.joblib')
mean = scaler_params['mean']
scale = scaler_params['scale']

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract features from URL parameters
        # timestamp = float(request.args.get('timestamp'))
        price = float(request.args.get('price'))
        rsi = float(request.args.get('rsi'))

        # Manually scale the features using the loaded scaler parameters
        # timestamp_scaled = (timestamp - mean[0]) / scale[0]
        price_scaled = (price - mean[0]) / scale[0]
        rsi_scaled = (rsi - mean[1]) / scale[1]

        # Prepare the input data as a scaled tensor
        input_data = torch.tensor([[price_scaled, rsi_scaled]], dtype=torch.float)

        # Add sequence_length dimension if needed, assuming sequence_length is 1 for simplicity
        input_data = input_data.unsqueeze(0)  # Now input_data should be [1, 1, 3]
            
        with torch.no_grad():
            logits = model(input_data)  # Model outputs logits
            probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
            predicted_class = (probabilities > 0.5).int()  # Classify as 1 if > 0.5 else 0
            prediction_str = predicted_class.item()  # Convert to Python scalar
        
        output_mapping = {0: 'SignalShort', 1: 'SignalLong'}
        prediction_str = output_mapping[prediction_str]
        

        return jsonify({'prediction': prediction_str})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)