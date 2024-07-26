from flask import Flask, request, render_template
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('knee_recovery_model.h5')
scaler = joblib.load('scaler.pkl')

def calculate_manual_score(input_data):
    weights = {
        'knee_ext_1_wk': 0.125, 'knee_ext_2_wk': 0.06328125, 'knee_ext_4_wk': 0.03203125, 'knee_ext_8_wk': 0.01640625, 'knee_ext_12_wk': 0.00859375, 'knee_ext_24_wk': 0.0046875,
        'knee_flx_1_wk': 0.125, 'knee_flx_2_wk': 0.06328125, 'knee_flx_4_wk': 0.03203125, 'knee_flx_8_wk': 0.01640625, 'knee_flx_12_wk': 0.00859375, 'knee_flx_24_wk': 0.0046875,
        'kin_180_acl_recon_4_wk': 0.0625, 'kin_180_acl_recon_8_wk': 0.0625, 'kin_180_acl_recon_12_wk': 0.0625, 'kin_180_acl_recon_24_wk': 0.0625,
        'kin_60_acl_recon_4_wk': 0.0625, 'kin_60_acl_recon_8_wk': 0.0625, 'kin_60_acl_recon_12_wk': 0.0625, 'kin_60_acl_recon_24_wk': 0.0625
    }
    
    input_values = np.array([input_data[col] for col in weights.keys()]).reshape(1, -1)
    
    # Normalize input values using the loaded scaler
    normalized_values = scaler.transform(input_values).flatten()
    
    # Calculate the manual score using normalized values
    manual_score = sum(normalized_values[i] * weight for i, weight in enumerate(weights.values()))
    
    return manual_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        input_data = [float(request.form[key]) for key in request.form.keys()]
        
        # Convert input data to NumPy array for model prediction
        input_array = np.array([input_data])
        
        # Predict using the machine learning model
        prediction = model.predict(input_array)
        predicted_score = prediction[0][0]
        
        # Calculate manual score
        input_dict = dict(zip(request.form.keys(), input_data))
        manual_score = calculate_manual_score(input_dict)
        
        # Render the results on the web page
        prediction_text = f'Predicted Recovery Score: {predicted_score:.3f}'
        manual_score_text = f'Manually Calculated Score: {manual_score:.3f}'
        
    except ValueError as e:
        # Handle the case where conversion to float fails
        prediction_text = f'Error in input data: {str(e)}'
        manual_score_text = ''
    except Exception as e:
        # Handle any other unforeseen errors
        prediction_text = f'An unexpected error occurred: {str(e)}'
        manual_score_text = ''
    
    return render_template('index.html', prediction_text=prediction_text, manual_score_text=manual_score_text)

if __name__ == '__main__':
    app.run(debug=True)
