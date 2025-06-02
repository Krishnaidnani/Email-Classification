---

# 📧 Email Classification API (with PII Masking + Unsupervised Learning)

This project provides a FastAPI-based Email Classification API that:

* 🔐 Masks Personally Identifiable Information (PII) using regex and NLP.
* 🧠 Classifies emails into categories like *Request*, *Incident*, *Change*, and *Problem* using unsupervised KMeans clustering.
* 🚀 Offers a REST endpoint for real-time predictions.

---

## 🔧 Features

* ✅ **PII Masking** – Uses regex and spaCy to detect and mask emails, phone numbers, names, and card info.
* ✅ **TF-IDF Vectorization** – Converts email text into numerical features.
* ✅ **KMeans Clustering** – Groups similar emails without labeled data.
* ✅ **FastAPI Server** – Easy to run and test with interactive Swagger UI.
* ✅ **Detailed Output** – Returns original email, masked content, identified PII, and predicted category.

---

## 📁 Project Structure

```
.
├── main.py                  # FastAPI server (entry point)
├── masking.py               # PII masking logic
├── classifier.py            # Model training script
├── email_kmeans.joblib      # Trained KMeans clustering model
├── tfidf_vectorizer.joblib  # TF-IDF vectorizer
├── clustered_emails.csv     # Output with predicted cluster labels
├── email_dataset.csv        # Input dataset with raw emails
└── README.md                # Project documentation
```

---

## 🚀 Quick Start

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

*Your `requirements.txt` should include:*

```txt
fastapi
uvicorn
scikit-learn
pandas
spacy
joblib
```

Then download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

### 2. 🧠 Train the Model

Ensure your `email_dataset.csv` contains a column named `email`.

```bash
python classifier.py
```

This will:

* Mask PII from emails
* Train a TF-IDF + KMeans model
* Save the trained model and vectorizer
* Generate `clustered_emails.csv` with predicted categories

---

### 3. 🌐 Run the API Server

```bash
python main.py
```

API will be available at:
➡️ [http://localhost:7860/docs](http://localhost:7860/docs) *(Swagger UI)*

---

## 📬 API Usage

### POST `/classify`

**Request Example:**

```json
{
  "input_email_body": "Hi, I need help with my account. My name is John Doe and my email is john@example.com."
}
```

**Response Example:**

```json
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
```

---

## 🧪 Example Cluster Mapping

Customize categories based on your trained model. Default mapping:

```python
CLUSTER_NAMES = {
    0: "Request",
    1: "Incident",
    2: "Change",
    3: "Problem"
}
```

---

## 📌 Notes

* PII masking enhances privacy and model generalization.
* Cluster labels can be renamed after reviewing contents in `clustered_emails.csv`.

---
