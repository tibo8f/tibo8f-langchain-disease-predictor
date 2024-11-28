"""
Main application for running the disease prediction API.

Launch the backend API: uvicorn app:app --reload
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from tools.disease_description_tool import disease_description
from tools.disease_precautions_tool import disease_precautions
from tools.find_nearest_doctor_tool import find_nearest_doctor
from tools.predict_disease_tool import predict_disease
from tools.utils import get_accepted_symptoms
from prompts import get_system_message

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration variables
model_path = "../models/disease_prediction_model.pkl"
encoder_path = "../models/label_encoder.pkl"
dataset_file_path = "../datasets/dataset.csv"
precaution_file_path = "../datasets/symptom_precaution.csv"

# In-memory storage for conversation histories
session_histories: Dict[str, List[Dict[str, str]]] = {}

def load_resources():
    """
    Load models, encoders, and datasets needed for the application.
    """
    global ia_model, label_encoder, accepted_symptoms, symptoms_list_str
    import joblib

    # Load ML model and label encoder
    ia_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Load accepted symptoms from the dataset
    accepted_symptoms, accepted_symptoms_with_spaces = get_accepted_symptoms(dataset_file_path)

    # Create a string representation of symptoms for system messages
    symptoms_list_str = ", ".join(accepted_symptoms_with_spaces)

def setup_langchain():
    """
    Configure LangChain with tools and agent system messages.
    """
    global langgraph_agent_executor

    # Initialize LangChain components
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Low temperature for deterministic results
    memory = MemorySaver()

    
    # Define system instructions for the LangChain agent
    system_message = get_system_message(symptoms_list_str)
    

    # Register tools and create LangChain agent
    tools = [predict_disease, disease_description, disease_precautions, find_nearest_doctor]
    langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

# Initialize resources and LangChain,
load_resources()
setup_langchain()

# API Models
class UserInputRequest(BaseModel):
    user_input: List[str]

@app.get("/")
def root():
    """
    Health check endpoint for the API.
    """
    return {"message": "Welcome to the disease prediction API!"}

@app.post("/predict_disease/")
def predict_disease_endpoint(request: UserInputRequest, session_id: str):
    """
    Predict the disease based on user input.

    Args:
        request (UserInputRequest): User-provided input.
        session_id (str): Unique session ID.

    Returns:
        dict: Predicted disease and conversation history.
    """
    try:
        # Initialize session history if not present
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