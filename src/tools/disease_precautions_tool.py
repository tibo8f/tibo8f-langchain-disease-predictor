import pandas as pd
from langchain_core.tools import tool

# Charger les datasets nécessaires
precaution_df = pd.read_csv('../datasets/symptom_precaution.csv')

# Transformer la colonne "Disease" pour uniformiser les noms
precaution_df["Disease"] = (
    precaution_df["Disease"]
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

@tool
def disease_precautions(disease: str) -> str:
    """
    Gives precautions to be taken for a specified disease.

    Args:
        disease (str): The name of the disease.

    Returns:
        str: The precautions to take or an error message if the disease is not found.
    """   
    # Rechercher la maladie dans le dataset
    disease_row = precaution_df[precaution_df["Disease"].str.lower() == disease.lower()]
    
    # Si la maladie est trouvée, renvoyer les précautions
    if not disease_row.empty:
        precautions = disease_row.iloc[0][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
        precautions = precautions.dropna().tolist()  # Supprimer les valeurs nulles
        return f"Precautions for the disease '{disease}' are : {', '.join(precautions)}"
    
    # Si la maladie n'est pas trouvée
    return f"No precautions found for disease '{disease}'."
