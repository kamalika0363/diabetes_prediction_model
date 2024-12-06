import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

class DiabetesPredictor:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.dataset_path = os.path.join(current_dir, '..', 'datasets', 'diabetes_prediction_dataset_male_female.csv')
        
        self.model_path = os.path.join(os.path.dirname(self.dataset_path), 'trained_model.joblib')
        self.scaler_path = os.path.join(os.path.dirname(self.dataset_path), 'scaler.joblib')
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        if self.load_model():
            print("Loaded pre-trained model successfully!")
        else:
            print("Training new model...")
            self.train_model()
            self.save_model()
    
    def save_model(self):
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("Model saved successfully!")
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.classifier = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def preprocess_data(self, data):
        try:
            df = data.copy()
            
            gender_mapping = {'Male': 0, 'Female': 1}
            df['gender'] = df['gender'].map(gender_mapping)
            
            smoking_history_mapping = {
                'no info': 0, 'current': 1, 'ever': 2,
                'former': 3, 'never': 4, 'not current': 5,
                'No Info': 0
            }
            df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)
            
            df = df.fillna(0)
            
            return df
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def train_model(self):
        try:
            print("Loading dataset...")
            columns_to_use = ['gender', 'age', 'hypertension', 'heart_disease', 
                            'smoking_history', 'bmi', 'HbA1c_level', 
                            'blood_glucose_level', 'diabetes']
            
            diabetes_dataset = pd.read_csv(
                self.dataset_path, 
                usecols=columns_to_use,
                encoding='utf-8'
            )
            
            print("Dataset loaded. Shape:", diabetes_dataset.shape)
            
            diabetes_dataset = self.preprocess_data(diabetes_dataset)
            
            X = diabetes_dataset.drop(columns='diabetes', axis=1)
            Y = diabetes_dataset['diabetes']
            
            print("Splitting data...")
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, stratify=Y, random_state=2
            )
            
            print("Fitting scaler...")
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            
            print("Training model...")
            self.classifier.fit(X_train_scaled, Y_train)
            
            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.classifier.score(X_test_scaled, Y_test)
            print(f"Model accuracy: {accuracy:.2f}")
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

    def predict(self, input_data):
        try:
            input_dict = {
                'gender': [float(input_data[0])],
                'age': [float(input_data[1])],
                'hypertension': [float(input_data[2])],
                'heart_disease': [float(input_data[3])],
                'smoking_history': [float(input_data[4])],
                'bmi': [float(input_data[5])],
                'HbA1c_level': [float(input_data[6])],
                'blood_glucose_level': [float(input_data[7])]
            }
            
            input_df = pd.DataFrame(input_dict)
            scaled_input = self.scaler.transform(input_df)
            prediction = self.classifier.predict(scaled_input)
            probability = self.classifier.predict_proba(scaled_input)[0]
            
            probability_percentage = probability[1] * 100 if prediction[0] == 1 else (1 - probability[1]) * 100
            
            return bool(prediction[0]), probability_percentage
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise