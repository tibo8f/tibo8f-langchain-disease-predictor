import pandas as pd
import re

def get_accepted_symptoms(file_path: str):
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
    def clean_symptom(symptom: str) -> str:
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
