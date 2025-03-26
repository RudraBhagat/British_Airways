from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and feature names
model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load feature names from training

# List of unique booking origins and routes from training
unique_countries = ['New Zealand', 'India', 'United Kingdom', 'China', 'South Korea',
                    'Japan', 'Malaysia', 'Singapore', 'Switzerland', 'Germany']
unique_routes = ['AKLDEL', 'AKLHGH', 'AKLHND', 'AKLICN', 'AKLKIX', 'AKLKTM', 'AKLKUL', 'AKLMRU', 'AKLPEK', 'AKLPVG']

def preprocess_input(user_input):
    """ Convert user input into the format expected by the model """
    user_df = pd.DataFrame([user_input])
    user_df = user_df.astype(float, errors='ignore')  # Convert numerical values

    # One-hot encode booking_origin
    user_df = pd.get_dummies(user_df)
    for country in unique_countries:
        column_name = f"booking_origin_{country}"
        if column_name not in user_df:
            user_df[column_name] = 0  # Add missing country columns with 0

    # One-hot encode route
    for route in unique_routes:
        column_name = f"route_{route}"
        if column_name not in user_df:
            user_df[column_name] = 0  # Add missing route columns with 0

    # Reorder columns to match training data
    user_df = user_df.reindex(columns=feature_names, fill_value=0)
    return user_df.to_numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form.to_dict()
        processed_input = preprocess_input(user_input)

        # Debugging: Check input shape
        print("Processed Input Shape:", processed_input.shape)
        print("Model Expected Features:", len(feature_names))

        prediction = model.predict(processed_input)[0]
        result = "Booking Completed" if prediction == 1 else "Booking Not Completed"
        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
