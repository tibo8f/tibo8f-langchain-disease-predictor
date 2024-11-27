# Comment lancer le projet ?

Créer un fichier .env au niveau du dossier mère dans lequel vous indiquez vos clés API. Vous devez fournir

- la clé API du modèle OPENAI (see https://openai.com/index/openai-api/ to create an OpenAI API Key)
- La clé API de votre compte Google Cloud. Créer une clé API avec un essai gratuit via https://cloud.google.com
- Si vous utilisez Langsmith pour tracer ce qui se passe, créer aussi un compte langsmith et une clé api langsmith via https://www.langchain.com/langsmith. Dans langsmith donnez un nom au projet qui correspond à ce que vous mettez dans le.env. Si vous n'utilisez pas langsmith retirer ces deux lignes de code dans app.py pour que le projet fonctionne:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
```

Votre .env doit ressembler à ça :

```.env
OPENAI_API_KEY="your_api_key"

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_api_key"
LANGCHAIN_PROJECT="name_of_the_project"

GOOGLE_MAP_API_KEY="your_api_key"
```

Il arrive que le modèle ia disease_prediction ne foncionne plus bien. Relancer le code entier dans disease_prediction_model.ipynb (le modèle n'est pas si long à recréer 30 secondes chez moi), vérifiez avant que les lignes de codes pour sauver le modèle ne sont pas en commentaire. Une fois terminé deux fichiers disease_prediction_model.pkl et label_encoder.pkl ont été créé, déplacez les dans le dossier models et supprimer les anciens fichiers

## Lancer le modèle

Ouvre deux terminal Launch the API app by running:

```bash
uvicorn app:app --reload
```

Et lance l'ui avec :

```bash
streamlit run ui.py
```
