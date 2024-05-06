from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the tuned RandomForestClassifier model
best_rf_model = joblib.load('new_rf_model.joblib')

# Load features data
features_data = joblib.load('features_data.joblib')

# Define the class labels
class_labels = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        input_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }

        # Create a DataFrame from the input data
        input_series = pd.Series(input_data, index=features_data['columns'])
        input_df = pd.DataFrame([input_series])

        # Make prediction using the trained model
        output = best_rf_model.predict(input_df)[0]

        # Get the recommended crop label
        recommended_crop = class_labels[output]

        return render_template('index.html', recommended_crop=recommended_crop)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
