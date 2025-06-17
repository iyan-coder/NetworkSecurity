### Network Security Projects For Phising Data

#  Network Security - Phishing Website Detection 🔍

A complete end-to-end Machine Learning project that detects phishing websites using supervised learning. The project leverages a clean pipeline architecture, experiment tracking with MLflow & DagsHub, and includes full model deployment.

---

## Problem Statement

With increasing digital activity, phishing websites are a major cybersecurity threat. This project aims to classify whether a website is **phishing** or **legitimate** using machine learning models trained on web-based features extracted from URLs.

---

##  Dataset

- **Name:** Phishing Website Detection Dataset  
- **Source:** [Kaggle - Phishing Website Detection](https://www.kaggle.com/datasets/sid321axn/phishing-website-detector)  
- **Format:** CSV  
- **Features:** 30+ web attributes (e.g., URL length, SSL state, domain age)  
- **Target Variable:** `Result`  
  - `-1`: Phishing  
  - `1`: Legitimate

---

## Project Structure

networksecurity/
├── data/
├── notebooks/
├── networksecurity/
│ ├── components/
│ ├── constant/
│ ├── entity/
│ ├── pipeline/
│ ├── utils/
│ ├── logger/
│ ├── exception/
│ └── app.py
├── config/
│ └── model.yaml
├── final_models/
├── mlruns/
├── requirements.txt
├── main.py
└── README.md


---

## Features & Stack

| Feature                    | Technology Used                 |
|---------------------------|----------------------------------|
| Data Handling             | `pandas`, `numpy`                |
| Modeling                  | `sklearn`, `xgboost`             |
| Tracking & Versioning     | `MLflow`, `DagsHub`              |
| Pipeline Architecture     | Modular OOP-based                |
| Deployment                | `FastAPI`, `Uvicorn`, `Docker` (Optional) |
| Model Evaluation          | Accuracy, F1-Score, ROC-AUC      |
| Storage                   | AWS S3 or DVC (Optional)         |
| Logging & Error Handling  | Custom logger, exception module  |

---

## ML Pipeline Stages

1. **Data Ingestion**
2. **Data Validation**
3. **Data Transformation**
4. **Model Training**
5. **Model Evaluation**
6. **Model Pushing (Saved Locally or to S3)**

---

## MLflow Integration

- Each pipeline step is tracked using **MLflow**
- Automatically logs parameters, metrics, and artifacts
- Integrated with [DagsHub](https://dagshub.com/) for remote experiment tracking

---

## How to Run

### Installation

```bash
git clone https://github.com/yourusername/networksecurity.git
cd networksecurity
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python main.py
mlflow ui
# Then open http://localhost:5000 in your browser
uvicorn app:app --reload
# Open browser: http://127.0.0.1:8000/docs
```

---

## Future Improvements

-  **Dockerization** – Containerize the entire project for consistent deployment  
-  **Unit Tests** – Add testing for each pipeline component and utility function  
-  **Model Explainability** – Integrate SHAP to explain feature importance  
-  **Streamlit Dashboard** – Build an interactive UI for real-time prediction and insights  

---

##  Author

**Adebayo Gabriel** – ML Engineer  
🔗 [GitHub](https://github.com/iyan-coder) | [LinkedIn](https://www.linkedin.com/in/gabriel-adebayo-2a0ba2281)

