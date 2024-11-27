"""
Tool for finding the nearest doctor based on a user's provided address using the Google Maps API.

To use this tool, ensure you have configured a valid Google Cloud API key.
1. Set up your API key:
   - Sign up for a free trial on Google Cloud: https://cloud.google.com.
   - Create an API key in the Google Cloud Console.
2. Add the key to your `.env` file in the following format:
   GOOGLE_MAP_API_KEY="your_api_key"
"""

import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

@tool
def find_nearest_doctor(address: str) -> str:
    """
    Finds the closest doctor based on a provided address.

    Args:
        address (str): The user's address as a string.

    Returns:
        str: Information about the nearest doctor, including their name, address, and phone number (if available).
    """
    # Retrieve the Google Maps API key from environment variables
    api_key = os.getenv("GOOGLE_MAP_API_KEY")
    if not api_key:
        return "Error: Google API key is not configured. Please set the GOOGLE_MAP_API_KEY environment variable."

    # Step 1: Convert the address to geographic coordinates (Geocoding)
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {
        "key": api_key,
        "address": address
    }

    geocode_response = requests.get(geocode_url, params=geocode_params)

    if geocode_response.status_code == 200:
        geocode_results = geocode_response.json().get("results", [])
        if geocode_results:
            # Extract latitude and longitude from the geocode response
            location = geocode_results[0]["geometry"]["location"]
            latitude, longitude = location["lat"], location["lng"]
        else:
            return "Unable to geocode the address. Please verify the input and try again."
    else:
        return f"Error during geocoding: {geocode_response.status_code}."

    # Step 2: Find the nearest doctor using Nearby Search
    nearby_search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    location_str = f"{latitude},{longitude}"  # Format location as 'latitude,longitude'

    search_params = {
        "key": api_key,
        "location": location_str,
        "rankby": "distance",  # Sort results by distance
        "type": "doctor"       # Filter results to doctors only
    }

    search_response = requests.get(nearby_search_url, params=search_params)

    if search_response.status_code == 200:
        search_results = search_response.json().get("results", [])
        if search_results:
            # Extract details of the closest doctor
            doctor = search_results[0]
            name = doctor.get("name", "Unknown Doctor")
            vicinity = doctor.get("vicinity", "Address not available")
            place_id = doctor.get("place_id", None)

            # Step 3: Retrieve additional details (e.g., phone number) using Place Details API
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

            # Return formatted doctor information
            return (
                f"The closest doctor is: {name}\n"
                f"Location: {vicinity}\n"
                f"Phone: {phone_number}"
            )
        else:
            return "No doctors found nearby. Please try a different location."
    else:
        return f"Error during Nearby Search: {search_response.status_code}."