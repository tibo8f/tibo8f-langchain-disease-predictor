"""streamlit run ui.py"""
import streamlit as st
import requests
import uuid

# Title of the application
st.title("Disease Prediction Based on Symptoms")

# Generate a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Initialize conversation history in the session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Input field for symptoms
symptoms = st.text_input("Enter your symptoms:")

# Predict button logic
if st.button("Predict"):
    try:
        # Send a POST request to the API with the session ID
        response = requests.post(
            f"http://127.0.0.1:8000/predict_disease/",
            json={"symptoms": [sym.strip() for sym in symptoms.split(",")]},
            params={"session_id": st.session_state["session_id"]},
        )

        # Check the response status
        if response.status_code == 200:
            data = response.json()
            # Update the local history with the new messages
            st.session_state["history"] = data["history"]
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Error during the request: {str(e)}")

# Display the conversation history
st.write("### Conversation History:")
for message in st.session_state["history"]:
    role = "User" if message["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {message['content']}")
