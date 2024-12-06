from flask import Flask, render_template, request, jsonify
from model.diabetes_model import DiabetesPredictor
import numpy as np
import traceback
import os

app = Flask(__name__)

def init_model():
    try:
        # Get the absolute path to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'datasets', 'diabetes_prediction_dataset_male_female.csv')
        
        print("="*50)
        print("Initializing model...")
        print(f"Looking for dataset at: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Checking alternative path...")
            # Try alternative path (for different project structures)
            dataset_path = os.path.join(current_dir, '..', 'datasets', 'diabetes_prediction_dataset_male_female.csv')
            print(f"Trying alternative path: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found in either location")

        # Initialize the model with the correct path
        model = DiabetesPredictor(dataset_path)
        print("Model initialized successfully!")
        print("="*50)
        return model
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        print("="*50)
        return None

# Initialize the model
predictor = init_model()

@app.route('/')
def home():
    global predictor
    if predictor is None:
        # Try to initialize the model again
        predictor = init_model()
        if predictor is None:
            return render_template('index.html', error="Model not initialized. Please check server logs.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predictor
    try:
        if predictor is None:
            # One last attempt to initialize the model
            predictor = init_model()
            if predictor is None:
                raise Exception("Model not initialized. Please check server logs.")

        # Get values from the form
        input_data = [
            float(request.form['gender']),
            float(request.form['age']),
            float(request.form['hypertension']),
            float(request.form['heart_disease']),
            float(request.form['smoking_history']),
            float(request.form['bmi']),
            float(request.form['hba1c_level']),
            float(request.form['blood_glucose_level'])
        ]
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        return render_template('result.html', 
                             prediction=prediction,
                             probability=probability,
                             input_data=input_data)
    
    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        traceback.print_exc()
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    if predictor is None:
        print("WARNING: Application starting without initialized model!")
        print("Please ensure the dataset file exists in one of these locations:")
        print("1.", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'diabetes_prediction_dataset_male_female.csv'))
        print("2.", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'diabetes_prediction_dataset_male_female.csv'))
    app.run(debug=True) 