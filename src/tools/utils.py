"""
Utility functions for processing datasets and extracting information.

This module provides helper functions for cleaning and standardizing symptom data.
"""

import pandas as pd
import re

def get_accepted_symptoms(file_path: str):
    """
    Extracts and processes unique symptoms from a dataset, ensuring consistency and standardization.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: 
            - A list of unique symptoms with underscores separating words (e.g., 'high_fever').
            - A corresponding list of symptoms with spaces instead of underscores (e.g., 'high fever').

    Example:
        Given a dataset column with values like " High Fever ", this function standardizes it to:
            - "high_fever" (underscore-separated)
            - "high fever" (space-separated)
    """
    # Load the dataset into a DataFrame
    dataset_df = pd.read_csv(file_path)

    # Helper function to clean and standardize symptom strings
    def clean_symptom(symptom: str) -> str:
        """
        Cleans and standardizes a symptom string by:
        - Stripping whitespace.
        - Converting to lowercase.
        - Replacing spaces with underscores.
        - Removing consecutive underscores.

        Args:
            symptom (str): The symptom string to clean.

        Returns:
            str: The cleaned symptom string.
        """
        symptom = symptom.strip().lower().replace(" ", "_")  # Lowercase and replace spaces with underscores
        symptom = re.sub(r"_+", "_", symptom)  # Replace multiple underscores with a single one
        return symptom

    # Clean and standardize each symptom column in the dataset
    for col in dataset_df.columns[1:]:  # Skip the first column (e.g., 'Disease' column)
        dataset_df[col] = dataset_df[col].fillna('').apply(
            lambda x: clean_symptom(str(x)) if x else ''  # Apply cleaning function, handle empty strings
        )

    # Collect all unique symptoms from the dataset
    unique_symptoms = set()
    for col in dataset_df.columns[1:]:
        unique_symptoms.update(dataset_df[col].unique())

    # Filter out empty strings from the unique symptoms
    standardized_symptoms = [symptom for symptom in unique_symptoms if symptom]

    # Generate a second list with spaces replacing underscores
    symptoms_with_spaces = [symptom.replace("_", " ") for symptom in standardized_symptoms]

    return standardized_symptoms, symptoms_with_spaces