"""
Tool for predicting diseases based on a list of user-provided symptoms.

This tool uses a pre-trained neural network model to predict potential diseases based on a binary vector
representation of symptoms. It ensures consistency with the training dataset by verifying and encoding
the provided symptoms. 

Prerequisites:
Ensure the following files are present:
   - Pre-trained model (`disease_prediction_model.pkl`)
   - Label encoder (`label_encoder.pkl`)
   - Symptom columns variable (`symptom_columns.pkl`)
   - Dataset file (`dataset.csv`)
"""
import numpy as np
import pandas as pd
import joblib
import pickle
from langchain_core.tools import tool
from tools.utils import get_accepted_symptoms

# Load the pre-trained model and label encoder
model_path = "../models/disease_prediction_model.pkl"
encoder_path = "../models/label_encoder.pkl"
ia_model = joblib.load(model_path)  # Trained disease prediction model
label_encoder = joblib.load(encoder_path)  # Label encoder for decoding disease labels

# Load the symptom columns used for encoding
with open('symptom_columns.pkl', 'rb') as f:
    symptom_columns = pickle.load(f)

# Retrieve the list of accepted symptoms from the dataset
file_path = "../datasets/dataset.csv"
accepted_symptoms, accepted_symptoms_with_spaces = get_accepted_symptoms(file_path)

@tool
def predict_disease(symptoms: list[str]) -> str:
    """
    Predicts a disease based on the user's provided symptoms using a pre-trained model.

    Args:
        symptoms (list[str]): A list of symptoms provided by the user (must match the accepted symptoms).

    Returns:
        str: Predicted disease and its certainty or an appropriate error message.
    """
    # Validate the symptoms by filtering out any unrecognized entries
    valid_symptoms = [symptom for symptom in symptoms if symptom in accepted_symptoms]

    if not valid_symptoms:
        return "These symptoms are not recognized."

    # Encode symptoms into a binary vector representing their presence
    encoded_vector = np.zeros(len(symptom_columns))
    for symptom in valid_symptoms:
        if symptom in symptom_columns:
            encoded_vector[symptom_columns.get_loc(symptom)] = 1

    try:
        # Predict probabilities for each disease
        probabilities = ia_model.predict(encoded_vector.reshape(1, -1))
        predicted_label = np.argmax(probabilities)  # Get the disease index with the highest probability
        certainty = probabilities[0][predicted_label]  # Get the certainty score of the prediction
        predicted_disease = label_encoder.inverse_transform([predicted_label])[0]  # Decode the label

        # If certainty is below 75%, suggest more symptoms
        if certainty < 0.75:
            return f"Ask for more symptoms because the prediction certainty is only {certainty * 100:.2f}%."
        
        return f"The predicted disease is: {predicted_disease} with a certainty of {certainty * 100:.2f}%."
    except Exception as e:
        # Catch and return any errors during prediction
        return f"Error during prediction: {str(e)}"
