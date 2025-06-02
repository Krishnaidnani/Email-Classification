from masking import mask_pii
import joblib
import json

# Load saved model and vectorizer
kmeans = joblib.load('email_kmeans.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

CLUSTER_NAMES = {
    0: "Request",
    1: "Incident",
    2: "Change",
    3: "Problem"
}

def classify_email_unsupervised(email_text):
    # Mask PII
    masked_email, entities = mask_pii(email_text)

    # Vectorize
    vec = vectorizer.transform([masked_email])

    # Predict cluster
    cluster = kmeans.predict(vec)[0]
    cluster_name = CLUSTER_NAMES.get(cluster, "Unknown")

    result = {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": cluster_name  # <-- Added human-readable cluster name here
    }

    return json.dumps(result, indent=2)

