import pandas as pd
from langchain_core.tools import tool

# Charger les datasets nécessaires
description_df = pd.read_csv('./datasets/symptom_Description.csv')

# Transformer la colonne "Disease" pour uniformiser les noms
description_df["Disease"] = (
    description_df["Disease"]
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

@tool
def disease_description(disease: str) -> str:
    """
    Gives a description of the specified disease.

    Args:
        disease (str): The name of the disease.

    Returns:
        str: The description of the disease or an error message if the disease is not found.
    """ 
    # Rechercher la maladie dans le dataset
    disease_row = description_df[description_df["Disease"].str.lower() == disease.lower()]
    
    # Si la maladie est trouvée, renvoyer la description
    if not disease_row.empty:
        description = disease_row.iloc[0]["Description"]
        return f"Description of the disease '{disease}' is : {description}"
    
    # Si la maladie n'est pas trouvée
    return f"Il n'y a pas de description pour la maladie '{disease}'."
