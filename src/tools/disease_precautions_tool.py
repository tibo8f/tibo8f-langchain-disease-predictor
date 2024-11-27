"""
Tool for retrieving recommended precautions for specific diseases based on a standardized dataset.
"""

import pandas as pd
from langchain_core.tools import tool

# Load the dataset containing disease precautions
try:
    precaution_df = pd.read_csv('../datasets/symptom_precaution.csv')
except FileNotFoundError:
    raise FileNotFoundError("The 'symptom_precaution.csv' file could not be found in the '../datasets/' directory.")

# Preprocess the "Disease" column to standardize disease names
precaution_df["Disease"] = (
    precaution_df["Disease"]
    .str.lower()  # Convert to lowercase for uniformity
    .str.strip()  # Remove leading/trailing spaces
    .str.replace(" ", "_")  # Replace spaces with underscores for consistency
)

@tool
def disease_precautions(disease: str) -> str:
    """
    Retrieve recommended precautions for a specified disease.

    Args:
        disease (str): The name of the disease (case-insensitive).

    Returns:
        str: A list of precautions for the disease, or an error message if the disease is not found.
    """
    # Standardize the input disease name to match dataset formatting
    standardized_disease = disease.strip().lower().replace(" ", "_")

    # Search for the disease in the dataset
    disease_row = precaution_df[precaution_df["Disease"] == standardized_disease]

    if not disease_row.empty:
        # Extract the precautions as a list, removing any null values
        precautions = disease_row.iloc[0][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
        precautions = precautions.dropna().tolist()

        # Return the precautions as a formatted string
        return f"Precautions for the disease '{disease}': {', '.join(precautions)}"

    # Return an error message if the disease is not found
    return f"No precautions found for the disease '{disease}'."
