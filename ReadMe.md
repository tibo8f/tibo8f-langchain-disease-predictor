# Comment lancer le projet ?

Pour faire fonctionner le projet tu auras besoin :

- une clé API de OpenAI, ou d'une autre service mais il faudra modifier certaines importations. (see https://openai.com/index/openai-api/ to create an OpenAI API Key)
- une clé api Google Cloud pour accéder au service api de google map. Tu peux créer une clé API Google Cloud en utilisant l'essai gratuit de 90 jours via https://cloud.google.com
- Optionnelement une clé langsmith pour inspecter ce que l'agent langchain fait. Créer un compte langsmith https://www.langchain.com/langsmith. Créer ensuite une clé. Dans langsmith donnez un nom au projet qui correspond à ce que vous mettez dans le.env. Si tu décides de ne pas utiliser langsmith retirer ces deux lignes de code dans `app.py` pour que le projet fonctionne:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
```

Créer .env au niveau du dossier mère dans lequel tu mets toutes ce clés API. Vous devez fournir
Votre .env doit ressembler à ça avec la clé OpenAI, la clé Google Cloud et les info langsmith :

```.env
OPENAI_API_KEY="your_api_key"

GOOGLE_MAP_API_KEY="your_api_key"

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_api_key"
LANGCHAIN_PROJECT="name_of_the_project"
```

## Lancer le modèle

Ouvre deux terminal Launch the API app by running:
Aller dans le dossier src et lancer les commandes :

```bash
uvicorn app:app --reload
```

Et lance l'ui avec :

```bash
streamlit run ui.py
```

# How to use the system

- you provide symptoms
- you can ask for the description or the precautions to take

## Relancer le modèle NN

Si la prédictions des maladies ne se fait pas bien il se peut que vous ayez à recréer le modèle. Pour ça aller dans le fichier disease_prediction.ipynb et exécuter l'ensemble des blocs de code pour réentrainer le modèle (prend 30 sec sur mac M1). Make sure que le dernier bloc de fichier réalisant la sauvegarde du modèle dans un fichier .pkl ne soit pas sous commentaire.

Une fois terminé deux fichiers disease_prediction_model.pkl et label_encoder.pkl ont été créé, déplacez les dans le dossier models et supprimer les anciens fichiers
