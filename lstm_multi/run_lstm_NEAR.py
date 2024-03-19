from flask import Flask, request, jsonify
import torch
import joblib
import json
from lstm_train import BiLSTMClassifierWithAttention

app = Flask(__name__)

coin = 'NEAR'

with open(f'models/{coin}_feature_names.json', 'r') as f:
    feature_names = json.load(f)
input_dims = len(feature_names)
    
model = BiLSTMClassifierWithAttention(input_dims, 32, 2, 3, 0.1)
model.load_state_dict(torch.load(f'models/{coin}_trained_model_lstm_300.pth', map_location=torch.device('cpu')))
model.eval()

# Load the saved scaler parameters
scaler_params = joblib.load(f'models/{coin}_scaler_params.joblib')
mean = scaler_params['mean']
scale = scaler_params['scale']

@app.route('/predict', methods=['GET'])
def predict():
    try:
        
        scaled_features = []
        
        for i, feature_name in enumerate(feature_names):
            # Extract feature value from URL parameters
            feature_value = float(request.args.get(feature_name))
            
            # Scale the feature
            feature_scaled = (feature_value - mean[i]) / scale[i]
            
            # Append the scaled feature to the list
            scaled_features.append(feature_scaled)

        input_data = torch.tensor([scaled_features], dtype=torch.float)
        # Add sequence_length dimension if needed, assuming sequence_length is 1 for simplicity
        input_data = input_data.unsqueeze(0)  # Now input_data should be [1, 1, 3]

        with torch.no_grad():
            prediction = model(input_data)
            _, predicted_class = torch.max(prediction, 1)
            prediction_str = predicted_class.item()
        
        output_mapping = {0: 'SignalShort', 1: 'SignalNone', 2: 'SignalLong'}
        prediction_str = output_mapping[prediction_str]
        

        return jsonify({'prediction': prediction_str})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)