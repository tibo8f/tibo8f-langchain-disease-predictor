import numpy as np
import pandas as pd
import joblib
import pickle
import re
from langchain_core.tools import tool
from tools.utils import get_accepted_symptoms

# Charger les modèles nécessaires
model_path = "./models/disease_prediction_model.pkl"
encoder_path = "./models/label_encoder.pkl"
ia_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Charger la variable symptom_columns pour garantir la correspondance des colonnes lors de l'encodage
with open('symptom_columns.pkl', 'rb') as f:
    symptom_columns = pickle.load(f)

# Charger la liste des symptômes acceptés

file_path = "./datasets/dataset.csv"
accepted_symptoms, accepted_symptoms_with_spaces = get_accepted_symptoms(file_path)

@tool
def predict_disease(symptoms: list[str]) -> str:
    """
    AI that predicts a disease based on a list of symptoms.

    Args:
        symptoms (list[str]): List of symptoms provided by the user.

    Returns:
        str: Predicted disease and certainty or an error message.
    """
    # Filter the input symptoms to keep only those that are valid
    valid_symptoms = [symptom for symptom in symptoms if symptom in accepted_symptoms]

    if not valid_symptoms:
        return "These symptoms are not recognized."

    # Prepare an encoded vector with binary values representing the presence of each symptom
    encoded_vector = np.zeros(len(symptom_columns))
    for symptom in valid_symptoms:
        if symptom in symptom_columns:
            encoded_vector[symptom_columns.get_loc(symptom)] = 1

    try:
        # Predict the probabilities of diseases
        probabilities = ia_model.predict(encoded_vector.reshape(1, -1))
        predicted_label = np.argmax(probabilities)
        certainty = probabilities[0][predicted_label]
        predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

        if certainty < 0.75:
            return f"Ask for more symptoms because the prediction certainty is only {certainty * 100:.2f}%."
        
        return f"The predicted disease is: {predicted_disease} with a certainty of {certainty * 100:.2f}%."
    except Exception as e:
        return f"Error during prediction: {str(e)}"
