import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

# Charger les variables d'environnement
load_dotenv()

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
