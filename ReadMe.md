# AI-Driven Disease Prediction System

This project is a disease prediction system built with **LangChain**, neural networks, and **Google Maps API**, designed to improve healthcare accessibility and efficiency. The system predicts diseases based on user-reported symptoms, provides descriptions and recommended precautions, and locates the nearest healthcare provider.

![Project Banner](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/banner.png)

---

## 📖 **Features**

1. **Disease Prediction**

   - Predicts diseases based on user-reported symptoms using a trained neural network.
   - Provides a confidence score for each prediction.

2. **Detailed Medical Information**

   - Descriptions of the diagnosed disease.
   - Precautions to take based on the disease.

3. **Find the Nearest Doctor**

   - Leverages Google Maps API to locate and recommend the closest doctor.
   - Provides contact details for healthcare professionals.

4. **Interactive User Interface**
   - Built with **Streamlit** for seamless interactions.

---

## 🚀 **Getting Started**

### **Requirements**

- Python >= 3.9
- OpenAI API Key for disease prediction: [Get one here](https://platform.openai.com/signup/).
- Google Cloud API Key for locating doctors: [Sign up for free trial](https://cloud.google.com/).
- Optional: LangSmith API Key for tracking agent interactions: [Learn more](https://www.langchain.com/langsmith).

### **Setup**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tibo8f/tibo8f-langchain-disease-predictor.git
   cd tibo8f-langchain-disease-predictor
   ```

2. **Install Dependencies**

   Use `pip` to install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add API Keys**  
   Create a `.env` file in the root directory and add your API keys:

   ```env
   OPENAI_API_KEY="your_openai_api_key"
   GOOGLE_MAP_API_KEY="your_google_map_api_key"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY="your_langsmith_api_key"
   LANGCHAIN_PROJECT="your_project_name"
   ```

   **Note:** If you choose not to use LangSmith, remove the following lines from `app.py`:

   ```python
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
   ```

---

## 🔧 **Usage**

### **Launching the Application**

1. Open two terminals.
2. In the first terminal, start the backend FastAPI service:

   ```bash
   cd src
   uvicorn app:app --reload
   ```

3. In the second terminal, start the user interface:

   ```bash
   streamlit run ui.py
   ```

4. Access the application in your browser: [http://localhost:8501](http://localhost:8501).

---

## 🛠 **Key Components**

### **1. Backend System**

The backend processes user inputs to predict diseases, fetch descriptions, recommend precautions, and locate nearby doctors.

- **`app.py`**: Main API logic.
- **`tools/`**: Modular tools for different tasks like prediction, description fetching, etc.

![Backend Architecture](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/backend_diagram.png)

### **2. User Interface**

The front-end is built using **Streamlit** to provide a simple and interactive user experience.

![Streamlit UI](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/ui_screenshot.png)

### **3. Neural Network Model**

The trained neural network predicts diseases with high accuracy.  
**Structure**:

- Input Layer: 132 nodes (symptoms).
- Hidden Layers: 128 and 64 neurons.
- Output Layer: 41 nodes (diseases).

Model files:

- `disease_prediction_model.pkl`
- `label_encoder.pkl`

---

## 🔄 **Updating the Model**

If disease predictions are inaccurate, retrain the neural network:

1. Open `disease_prediction_model.ipynb` in your Jupyter Notebook or Colab.
2. Run all the cells to preprocess data and retrain the model.
3. Save the updated model (`disease_prediction_model.pkl` and `label_encoder.pkl`).
4. Replace the old files in the `/models` directory.

---

## 🖼 **Screenshots**

### **Disease Prediction**

![Disease Prediction](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/disease_prediction.png)

### **Finding the Nearest Doctor**

![Find Doctor](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/find_doctor.png)

---

## 📊 **Dataset**

The system uses a dataset with 132 symptoms and 41 diseases.  
Source: [Kaggle Disease Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

---

## 🌿 **Environmental Impact**

- **Minimal Training Energy**: Training uses just 16 seconds on a Mac M1, consuming ~3.11 mg CO₂.
- **Google Maps API**: Each query emits ~19.17 mg CO₂.
- **OpenAI API**: Each call emits ~4.32g CO₂.

![Energy Usage](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/environmental_impact.png)

---

## 🔍 **Troubleshooting**

### Common Issues

1. **KeyError for symptoms**: Ensure user inputs match symptom names in the dataset.
2. **Missing .env file**: Make sure API keys are added to the `.env` file.
3. **Streamlit Errors**: Check Python version and ensure all dependencies are installed.

### Contact

For issues, create a GitHub issue or email `your-email@example.com`.

---

## 🙌 **Contributions**

Contributions are welcome! Please open an issue or submit a pull request.

---

## 🏆 **Acknowledgments**

This project supports **Sustainable Development Goal (SDG) 3: Good Health and Well-being**, aiming to reduce inequalities in healthcare access.

![SDG 3](https://github.com/tibo8f/tibo8f-langchain-disease-predictor/assets/sdg3.png)

- Built by **Thibaut François** as part of a Master’s AI Project.
- Special thanks to [LangChain](https://www.langchain.com/) for their powerful tools.
