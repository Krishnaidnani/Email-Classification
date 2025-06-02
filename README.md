# 📧 Email Classification API (with PII Masking + Unsupervised Learning)

This project implements an Email Classification API that:
- 🔐 Detects and masks sensitive Personally Identifiable Information (PII) using regular expressions and spaCy NLP.
- 📊 Classifies emails into categories such as _Request_, _Incident_, _Change_, and _Problem_ using unsupervised machine learning (KMeans).
- 🚀 Provides a FastAPI-powered REST endpoint for real-time predictions.

---

## 🔧 Features

- ✅ PII Masking: Regex + NLP-based detection of emails, phone numbers, names, card info, etc.
- ✅ TF-IDF Vectorization for text processing.
- ✅ KMeans Clustering for unsupervised categorization.
- ✅ FastAPI server for deployment and testing.
- ✅ Outputs include original email, masked entities, and predicted category.

---

## 📁 Project Structure

Bash

.
├── main.py                  # FastAPI server
├── masking.py              # PII masking logic
├── classifier.py          # Model training script
├── email_kmeans.joblib     # Saved clustering model (generated after training)
├── tfidf_vectorizer.joblib # Saved TF-IDF vectorizer (generated after training)
├── clustered_emails.csv    # Output file with cluster labels
├── email_dataset.csv       # Input dataset (user provided)
└── README.md               # This file

---

## 🚀 Quick Start

### 1. 📦 Install Dependencies

Bash

pip install -r requirements.txt

_requirements.txt should include:_

txt

fastapi
uvicorn
scikit-learn
pandas
spacy
joblib

Then install the spaCy model:

Bash

python -m spacy download en_core_web_sm

---

### 2. 🧠 Train the Model

Ensure your email_dataset.csv has a column named email:

Bash

python train_model.py

This will:
- Mask PII
- Train a KMeans model
- Save the model + vectorizer
- Export clustered_emails.csv with cluster labels

---

### 3. 🌐 Run the API

Bash

python api.py

Visit the interactive API docs at:  
[http://localhost:7860/docs](http://localhost:7860/docs)

---

## 📬 API Usage

### POST /classify

Request JSON:
JSON

{
  "input_email_body": "Hi, I need help with my account. My name is John Doe and my email is john@example.com."
}

Response JSON:
JSON

{
  "input_email_body": "Hi, I need help with my account. My name is John Doe and my email is john@example.com.",
  "list_of_masked_entities": [
    {
      "position": [42, 50],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [69, 86],
      "classification": "email",
      "entity": "john@example.com"
    }
  ],
  "masked_email": "Hi, I need help with my account. My name is [full_name] and my email is [email].",
  "category_of_the_email": "Request"
}

---

## 🧪 Example Clusters

Customize the cluster names as per your KMeans results. Default categories:

Python

CLUSTER_NAMES = {
    0: "Request",
    1: "Incident",
    2: "Change",
    3: "Problem"
}

---

## 📌 Notes

- Masked data improves generalization and reduces bias in clustering.
- This system uses unsupervised clustering, so cluster names can be adjusted after analyzing email themes in clustered_emails.csv.

---