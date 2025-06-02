from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

from masking import mask_pii

# Load models
kmeans = joblib.load("email_kmeans.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Optional: Map clusters to labels (update after reviewing clusters)
cluster_to_category = {
    0: "Incident",
    1: "Request",
    2: "Change",
    3: "Problem"
}

# FastAPI app
app = FastAPI(title="Email Classification API", version="1.0")


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Email Classification API is running."}


# Input schema
class EmailRequest(BaseModel):
    input_email_body: str


@app.post("/classify")
def classify_email(request: EmailRequest):
    """
    Accepts an email, masks PII, classifies it into a support category,
    and returns detailed structured output.
    """
    email_text = request.input_email_body

    # PII Masking
    masked_email, entities = mask_pii(email_text)

    # Vectorize and predict
    X_vec = vectorizer.transform([masked_email])
    cluster = kmeans.predict(X_vec)[0]
    category = cluster_to_category.get(cluster, f"Cluster {cluster}")

    # Final response
    response = {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

    return response


# Run with: python api.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
