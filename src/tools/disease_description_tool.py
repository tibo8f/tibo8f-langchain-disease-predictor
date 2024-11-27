"""
Tool for retrieving descriptions of diseases based on a standardized dataset.
"""

import pandas as pd
from langchain_core.tools import tool

# Load the dataset for disease descriptions
try:
    description_df = pd.read_csv('../datasets/symptom_Description.csv')
except FileNotFoundError:
    raise FileNotFoundError("The 'symptom_Description.csv' file could not be found in the '../datasets/' directory.")

# Preprocess the "Disease" column to standardize disease names
description_df["Disease"] = (
    description_df["Disease"]
    .str.lower()  # Convert to lowercase for uniformity
    .str.strip()  # Remove leading/trailing spaces
    .str.replace(" ", "_")  # Replace spaces with underscores for consistency
)

@tool
def disease_description(disease: str) -> str:
    """
    Retrieve the description of a specified disease.

    Args:
        disease (str): The name of the disease (case-insensitive).

    Returns:
        str: The description of the disease, or an error message if the disease is not found.
    """
    # Ensure the disease name is preprocessed in the same way as the dataset
    standardized_disease = disease.strip().lower().replace(" ", "_")

    # Search for the disease in the dataset
    disease_row = description_df[description_df["Disease"] == standardized_disease]
    
    if not disease_row.empty:
        # Extract the description from the first matching row
        description = disease_row.iloc[0]["Description"]
        return f"Description of the disease '{disease}': {description}"
    
    # Return an error message if the disease is not found
    return f"No description found for the disease '{disease}'."
