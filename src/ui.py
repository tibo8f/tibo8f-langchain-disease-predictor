"""
Streamlit-based User Interface for LangChain Disease Prediction System.

Launch the app using the command: `streamlit run ui.py`

This UI allows users to interact with the LangChain-based assistant by entering any input text.
The application communicates with a FastAPI backend for processing and maintains a session-based
conversation history to ensure a seamless user experience.
"""

import streamlit as st
import requests
import uuid

# --- Application Title ---
st.title("Health Assistant")

# --- Session Management ---
# Generate a unique session ID for each user (used for backend session tracking)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Initialize conversation history in the session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- User Input Section ---
# Input field for the user to enter text
user_input = st.text_input("Enter your symptoms to find out what disease you might have, or provide your address to locate the nearest doctor :")

# --- Predict Button Logic ---
if st.button("Send"):
    try:
        # Send the user's input to the FastAPI backend
        response = requests.post(
            url="http://127.0.0.1:8000/predict_disease/",
            json={"user_input": [user_input.strip()]},
            params={"session_id": st.session_state["session_id"]},
        )

        # Process the API response
        if response.status_code == 200:
            data = response.json()
            # Update the session history with new messages from the backend
            st.session_state["history"] = data["history"]
        else:
            # Display an error message if the backend returns an error
            st.error(f"Error from API: {response.text}")
    except Exception as e:
        # Handle exceptions that might occur during the API request
        st.error(f"Error during the request: {str(e)}")

# --- Display Conversation History ---
# Display the conversation history between the user and the assistant
st.write("### Conversation History:")
for message in st.session_state["history"]:
    # Distinguish between user and assistant roles in the conversation
    role = "User" if message["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {message['content']}")


