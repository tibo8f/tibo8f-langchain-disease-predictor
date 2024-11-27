"""
lauch the api app by running : uvicorn app:app --reload
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import re

import requests

# Charger les variables d'environnement
load_dotenv()

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialisation
app = FastAPI()


# Charger les modèles et les données
model_path = "./models/disease_prediction_model.pkl"
encoder_path = "./models/label_encoder.pkl"
description_df = pd.read_csv('./models/symptom_Description.csv')
precaution_df = pd.read_csv('./models/symptom_precaution.csv')
# Transform the 'Disease' column to be standarised
precaution_df["Disease"] = precaution_df["Disease"].str.lower().str.strip().str.replace(" ", "_")
description_df["Disease"] = description_df["Disease"].str.lower().str.strip().str.replace(" ", "_")

def get_accepted_symptoms(file_path):
    """
    Process the dataset to clean symptoms and extract unique symptoms.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        tuple: A sorted list of unique symptoms with underscores and the same list with spaces instead.
    """
    # Load the dataset
    dataset_df = pd.read_csv(file_path)

    # Function to clean a symptom string
    def clean_symptom(symptom):
        # Remove leading/trailing whitespace, convert to lowercase, replace spaces with underscores
        symptom = symptom.strip().lower().replace(" ", "_")
        # Replace multiple underscores with a single underscore
        symptom = re.sub(r"_+", "_", symptom)
        return symptom

    # Clean symptom columns
    for col in dataset_df.columns[1:]:  # Skip the first column if it's 'Disease' or similar
        dataset_df[col] = dataset_df[col].fillna('').apply(
            lambda x: clean_symptom(str(x)) if x else ''
        )

    # Extract all unique symptoms from the dataset
    unique_symptoms = set()
    for col in dataset_df.columns[1:]:
        unique_symptoms.update(dataset_df[col].unique())

    # Remove any empty strings 
    accepted_symptoms = [symptom for symptom in unique_symptoms if symptom]

    # Create a list with symptoms using spaces instead of underscores
    accepted_symptoms_with_spaces = [symptom.replace("_", " ") for symptom in accepted_symptoms]

    return accepted_symptoms, accepted_symptoms_with_spaces

# Usage
file_path = "./models/dataset.csv"
accepted_symptoms, accepted_symptoms_with_spaces = get_accepted_symptoms(file_path)

symptoms_list_str = ", ".join(accepted_symptoms_with_spaces)




ia_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)



# Configurer LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = MemorySaver()


import pickle

# Charger la variable symptom_columns depuis le fichier où le modèle à été créé pour garder le même format d'encodage binaire. sinon les symptomes ne s'écrivent pas dans le même ordre lors de la création du modèle et lors de son utilisation. Si la matrice binaire est créé avec l'ordre des colonnes différent de ce la matrice binaire qu'on utilise. Les colonnes de symptomes ne correspondent pas et les symptoms sont mélangés.
with open('symptom_columns.pkl', 'rb') as f:
    symptom_columns = pickle.load(f)

@tool
def predict_disease(symptoms: list[str]) -> str:
    """
    AI that predicts a disease based on a list of symptoms.
    """
    # Filter the input symptoms to keep only those that are valid (present in the accepted symptoms list)
    valid_symptoms = [symptom for symptom in symptoms if symptom in accepted_symptoms]

    # If no valid symptoms are found, return an appropriate message
    if not valid_symptoms:
        return "These symptoms are not recognized."
    
    # Prepare an encoded vector with binary values representing the presence of each symptom
    encoded_vector = np.zeros(len(symptom_columns))
    for symptom in valid_symptoms:
        if symptom in symptom_columns:
            encoded_vector[symptom_columns.get_loc(symptom)] = 1

    try:
        # Predict the probabilities of diseases based on the encoded symptoms vector
        probabilities = ia_model.predict(encoded_vector.reshape(1, -1))
        predicted_label = np.argmax(probabilities)  # Find the index of the highest probability
        certainty = probabilities[0][predicted_label]  # Extract the certainty of the prediction
        predicted_disease = label_encoder.inverse_transform([predicted_label])[0]  # Decode the disease label

        # If the certainty is below 75%, request more symptoms to improve accuracy
        if certainty < 0.75:
            return (f"Ask for more symptoms because de prediction certainty is only {certainty * 100:.2f}%. ")
        
        # Return the predicted disease along with its certainty
        return f"The predicted disease is: {predicted_disease} with a certainty of {certainty * 100:.2f}%."
    except Exception as e:
        # Handle any errors that occur during prediction and return an error message
        return f"Error during prediction: {str(e)}"


@tool
def disease_description(disease: str) -> str:
    """
    Gives a description of the specified disease.

    Args:
        disease (str): The name of the disease.

    Returns:
        str: The description of the disease or an error message if the disease is not found.
    """ 
    # Rechercher la maladie dans le dataset
    disease_row = description_df[description_df["Disease"].str.lower() == disease.lower()]
    
    # Si la maladie est trouvée, renvoyer la description
    if not disease_row.empty:
        description = disease_row.iloc[0]["Description"]
        return f"Description of the disease '{disease}' is : {description}"
    
    # Si la maladie n'est pas trouvée
    return f"Il n'y a pas de description pour la maladie '{disease}'."

@tool
def disease_precautions(disease: str) -> str:
    """
    Gives precautions to be taken for a specified disease.

    Args:
        disease (str): The name of the disease.

    Returns:
        str: The precautions to take or an error message if the disease is not found.
    """   
    # Rechercher la maladie dans le dataset
    disease_row = precaution_df[precaution_df["Disease"].str.lower() == disease.lower()]
    
    # Si la maladie est trouvée, renvoyer les précautions
    if not disease_row.empty:
        precautions = disease_row.iloc[0][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
        precautions = precautions.dropna().tolist()  # Supprimer les valeurs nulles
        return f"Precautions for the disease '{disease}' are : {', '.join(precautions)}"
    
    # Si la maladie n'est pas trouvée
    return f"No precautions found for disease '{disease}'."

@tool
def find_nearest_doctor(address: str) -> str:
    """
    Finds the closest doctor based on a provided address.

    Args:
        address (str): The user's address.

    Returns:
        str: Information about the nearest doctor, including name, address, and phone number (if available).
    """
    # Retrieve the API key from environment variables
    api_key = os.getenv("GOOGLE_MAP_API_KEY")
    if not api_key:
        return "Error: Google API key is not configured."

    # Step 1: Geocode the Address to Get Coordinates
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {
        "key": api_key,
        "address": address
    }

    geocode_response = requests.get(geocode_url, params=geocode_params)
    if geocode_response.status_code == 200:
        geocode_results = geocode_response.json().get("results")
        if geocode_results:
            location = geocode_results[0]["geometry"]["location"]
            latitude = location["lat"]
            longitude = location["lng"]
        else:
            return "Unable to geocode the address. Please check the address and try again."
    else:
        return f"Error during geocoding: {geocode_response.status_code}"

    # Step 2: Use the Coordinates to Find the Nearest Doctor
    nearby_search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    location_str = f"{latitude},{longitude}"  # Build the location string

    # Define parameters for the Nearby Search API
    search_params = {
        "key": api_key,
        "location": location_str,
        "rankby": "distance",  # Rank results by proximity
        "type": "doctor"       # Filter results for doctors
    }

    # Call the Nearby Search API
    search_response = requests.get(nearby_search_url, params=search_params)

    if search_response.status_code == 200:
        search_results = search_response.json().get("results", [])
        if search_results:
            # Extract the first doctor's details
            doctor = search_results[0]
            name = doctor.get("name", "Unknown Doctor")
            vicinity = doctor.get("vicinity", "Address not available")
            place_id = doctor.get("place_id", None)

            # Get the phone number using Place Details API
            if place_id:
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "key": api_key,
                    "place_id": place_id,
                    "fields": "formatted_phone_number"
                }
                details_response = requests.get(details_url, params=details_params)
                if details_response.status_code == 200:
                    details_result = details_response.json().get("result", {})
                    phone_number = details_result.get("formatted_phone_number", "Phone number not available")
                else:
                    phone_number = "Unable to retrieve phone number."
            else:
                phone_number = "No additional details available."

            return (f"The closest doctor is: {name}, located at {vicinity}. "
                    f"Phone: {phone_number}.")
        else:
            return "No doctors found nearby."
    else:
        return f"Error during search: {search_response.status_code}"


tools = [predict_disease, disease_description, disease_precautions, find_nearest_doctor]


system_message = """
    You are a helpful assistant trained to assist users with health-related queries. Your primary goals are as follows:

    ### Goal 1: Predict Diseases Based on Symptoms
    - You have to predict diseases using the "predict_disease" tool. You can't predict a disease without calling the tool and you can't give certainty without calling the tool. You have to call the tool to give those informations.
    - The tool can only be called by providing symptoms from the list: {symptoms_list_str}. Symptoms not in this list are invalid.
    - If the user provides a symptom that does not exactly match the list, you must transform it to the closest corresponding symptom in {symptoms_list_str}. For instance:
        - "out of breath" → "breathlessness"
        - "High fever" or "Strong fever" → "high_fever"
    - Do not call the tool unless all symptoms in the input are valid and match the list after transformation. Do not ask user confirmation.

    - Once a disease is predicted:
      1. Inform the user of the predicted disease.
      2. Use the "disease_description" tool to provide a description of the predicted disease.
      3. Use the "disease_precautions" tool to suggest precautions for the predicted disease.

    - If no disease is found or the confidence level of prediction is too low, politely inform the user and ask for additional symptoms to improve the prediction.

    ### Goal 2: Locate the Nearest Doctor
    - Ask the user to provide their location to locate the nearest doctor.
    - Use the "find_nearest_doctor" tool to locate the nearest doctor.
    - Provide the user with the doctor's name and address.

    ### Guidelines for Using the Tools:
    1. **predict_disease**:
    - Input: A list of symptoms provided by the user. Symptoms must be written in lowercase and with underscores instead of spaces (e.g., "loss of appetite" → "loss_of_appetite"). All symptoms must belong to the list: {symptoms_list_str}.
    - Output: The predicted disease or a suggestion to provide more symptoms for better accuracy.

    2. **disease_description**:
    - Input: The name of the disease predicted by the "predict_disease" tool.
    - Output: A description of the disease.

    3. **disease_precautions**:
    - Input: The name of the disease predicted by the "predict_disease" tool.
    - Output: Precautionary measures for the disease.

    4. **find_nearest_doctor**:
    - Input: address (str): The user's address.
    - Output: The nearest doctor's name and address.
    - If no doctor is found, inform the user politely.

    ### Examples of Interaction:
    1. User provides symptoms: "I have a  loss of balance, Headache, and I am tired."
    - Transform symptoms to match the list: "loss of balance, Headache, tired" → ["loss_of_balance", "headache", "fatigue"]
    - Use "predict_disease" to predict the disease and inform the user with the result.
    - If a disease is predicted, inform the user and use "disease_description" and "disease_precautions" to give detailed information. Then, ask the user for their location to suggest the nearest doctor using "find_nearest_doctor."
    - If a disease is not predicted, ask the user for more symptoms AND their location to suggest the nearest doctor.

    2. User says: "Find a doctor near me."
    - Use "find_nearest_doctor" to locate the nearest doctor and provide their information.

    3. User asks: "What symptoms have I already shared?"
    - Respond with a list of the symptoms the user has shared so far.

    ### Additional Notes:
    - If a tool encounters an error (e.g., invalid inputs, no results), notify the user with a polite explanation.
    - Be concise and clear in your responses to the user.
    - Always adapt your responses to the language used in the user's query.
"""
langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

# Endpoint de prédiction
class SymptomsRequest(BaseModel):
    symptoms: list[str]

from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List

# Ajout d'un middleware pour autoriser les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage temporaire en mémoire pour les conversations
session_histories: Dict[str, List[Dict[str, str]]] = {}

# Endpoint de prédiction avec gestion d'historique
@app.post("/predict_disease/")
def predict_disease_endpoint(request: SymptomsRequest, session_id: str):
    try:
        # Récupérer l'historique pour cette session, ou initialiser s'il n'existe pas
        if session_id not in session_histories:
            session_histories[session_id] = []

        # Ajouter le nouveau message utilisateur à l'historique
        session_histories[session_id].append({"role": "user", "content": f"{', '.join(request.symptoms)}"})

        # Construire les messages à partir de l'historique
        messages = session_histories[session_id]

        # Obtenir la réponse de l'agent LangChain
        response = langgraph_agent_executor.invoke(
            {"messages": messages},
            {"configurable": {"thread_id": session_id, "checkpoint_ns": "default"}}
        )
        # Ajouter la réponse du chatbot à l'historique
        session_histories[session_id].append({"role": "assistant", "content": response["messages"][-1].content})

        return {"prediction": response["messages"][-1].content, "history": session_histories[session_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint de test
@app.get("/")
def root():
    return {"message": "Bienvenue dans l'application de prédiction de maladies basée sur des symptômes."}
