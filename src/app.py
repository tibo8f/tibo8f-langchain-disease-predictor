"""
Main application for running the disease prediction API.

Launch the app with: uvicorn app:app --reload
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

import pandas as pd
import joblib

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from tools.disease_description_tool import disease_description
from tools.disease_precautions_tool import disease_precautions
from tools.find_nearest_doctor_tool import find_nearest_doctor
from tools.predict_disease_tool import predict_disease
from tools.utils import get_accepted_symptoms

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware to allow communication from Streamlit or external tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load paths for models and datasets
model_path = "../models/disease_prediction_model.pkl"
encoder_path = "../models/label_encoder.pkl"
dataset_file_path = "../datasets/dataset.csv"
precaution_file_path = "../datasets/symptom_precaution.csv"

# Load model and encoder
ia_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Load accepted symptoms from the dataset
accepted_symptoms, accepted_symptoms_with_spaces = get_accepted_symptoms(dataset_file_path)

# Create a string representation of symptoms for system messaging
symptoms_list_str = ", ".join(accepted_symptoms_with_spaces)

# LangChain setup
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)          # temperature control the randomness of the response. Low temperature makes the model deterministic.
memory = MemorySaver()

# Register tools with LangChain
tools = [predict_disease, disease_description, disease_precautions, find_nearest_doctor]

# Define system instructions for the LangChain agent
system_message = f"""
    You are a helpful assistant trained to assist users with health-related queries. Your primary goals are as follows:

    ### Goal 1: Predict Diseases Based on Symptoms
    - You have to predict diseases using the "predict_disease" tool. You can't predict a disease without calling the tool and you can't give certainty without calling the tool. You have to call the tool to give those informations.
    - The tool can only be called by providing symptoms from the list: {symptoms_list_str}. Symptoms not in this list are invalid.
    - If the user provides a symptom that does not exactly match the list, you must transform it to the closest corresponding symptom in {symptoms_list_str}. For instance:
        - "out of breath" → "breathlessness"
        - "High fever" or "Strong fever" → "high_fever"
    - Do not call the tool unless all symptoms in the input are valid and match the list after transformation. Do not ask user confirmation.

    - Once a disease is predicted:
      1. Inform the user of the predicted disease with the certainty.
      2. Use the "disease_description" tool to provide a description of the predicted disease. Inform the user of the description.
      3. Use the "disease_precautions" tool to suggest precautions for the predicted disease. Inform the user of the precautions.

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
    - If a disease is predicted, inform the user of the disease he has and the certainty. Then use "disease_description" and "disease_precautions" to provide the description and the precautions. Then, ask the user for their location to suggest the nearest doctor using "find_nearest_doctor."
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

# Create the LangChain agent executor
langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

# Class for incoming prediction requests
class UserInputRequest(BaseModel):
    user_input: List[str]

# In-memory conversation storage
session_histories: Dict[str, List[Dict[str, str]]] = {}

# API Routes
@app.get("/")
def root():
    """
    Health check endpoint for the API.
    """
    return {"message": "Welcome to the disease prediction API!"}

@app.post("/predict_disease/")
def predict_disease_endpoint(request: UserInputRequest, session_id: str):
    """
    Predict the disease based on symptoms provided by the user.

    Args:
        request (SymptomsRequest): User-provided symptoms.
        session_id (str): Unique identifier for the user session.

    Returns:
        dict: Predicted disease and conversation history.
    """
    try:
        # Initialize session history if not already present
        if session_id not in session_histories:
            session_histories[session_id] = []

        # Add user input to session history
        session_histories[session_id].append({"role": "user", "content": ", ".join(request.user_input)})

        # Construct messages from session history
        messages = session_histories[session_id]

        # Get LangChain agent response
        response = langgraph_agent_executor.invoke(
            {"messages": messages},
            {"configurable": {"thread_id": session_id, "checkpoint_ns": "default"}},
        )

        # Append assistant's response to session history
        session_histories[session_id].append({"role": "assistant", "content": response["messages"][-1].content})

        # Return prediction and conversation history
        return {"prediction": response["messages"][-1].content, "history": session_histories[session_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))