# NEOLUNG

**An experimental AI-assisted project for exploring non-small-cell lung cancer (NSCLC) image and clinical-data classification.** This repository includes a Django web application, sample image and clinical datasets, and a question-and-answer chatbot.

> **Medical disclaimer:** NEOLUNG is an educational/research prototype, not a medical device. Its predictions and chatbot responses are not medical advice and must not be used to diagnose, treat, or make healthcare decisions. Consult a qualified healthcare professional.

## What is included

- **CT image classification:** the code defines a MobileNetV2 transfer-learning model and a web workflow for uploading an image and displaying a predicted class and confidence.
- **Clinical-data classification:** a separate prediction workflow accepts clinical features. The code expects a trained model and preprocessing artifacts.
- **NSCLC information chatbot:** a TF-IDF similarity-based question-and-answer chatbot backed by a CSV dataset. It is a retrieval chatbot, not a medical professional or a generative AI system.
- **Sample data:** image folders for benign, malignant, and normal cases, plus clinical train/test CSV files.
- **Django site:** registration, login, dashboard, prediction pages, and chatbot page.

## Repository layout

```text
.
├── dataset/
│   ├── Bengin cases/                 # Folder name as currently stored in the repository
│   ├── Malignant cases/
│   ├── Normal cases/
│   ├── clinical_dataset_train.csv
│   └── clinical dataset_test.csv
├── nsclc/
│   ├── nsclc/                        # Django project settings and URLs
│   ├── nsclc_app/                    # Web app, prediction code, and chatbot
│   ├── templates/                    # HTML pages
│   ├── static/                       # CSS, JavaScript, and images
│   ├── media/                        # Uploaded files and expected model artifacts
│   └── manage.py
└── nsclc_chatbot_dataset_60.csv
```

## Run locally

The project currently has no pinned dependency file (`requirements.txt`). It uses Python/Django and code imports TensorFlow, pandas, NumPy, scikit-learn, Pillow, and joblib. The Django settings also use a local MySQL database. Install versions compatible with your Python environment, then:

1. Create and activate a Python virtual environment.
2. Install the dependencies listed above, including a MySQL driver supported by Django.
3. Configure the MySQL connection and Django settings in `nsclc/nsclc/settings.py` for your local environment. The checked-in settings are development settings, not suitable for production.
4. From the `nsclc/` directory, run database migrations and start the development server:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

5. Open `http://127.0.0.1:8000/` in your browser.

### Model and chatbot setup

The web app expects trained artifacts under `nsclc/media/models/`: `nsclc_cnn.h5` for image prediction, and `nsclc_ann.h5`, `ann_scaler.pkl`, and `ann_encoders.pkl` for clinical prediction. These files are not included in the repository, so the corresponding prediction features require compatible trained artifacts before they can return predictions.

The chatbot code currently refers to a machine-specific CSV path. Update that path to the included `nsclc_chatbot_dataset_60.csv` (or another valid dataset path) before using the chatbot on a different machine.

## Data and responsible use

The sample files are provided for project exploration. Verify their source, permissions, and suitability before redistribution or research use. Model performance, data provenance, and clinical validity are not established by this repository.
