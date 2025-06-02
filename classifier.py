import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

from masking import mask_pii


def train_unsupervised_model(data_path, n_clusters=4):
    """
    Train a KMeans model on masked email data and save the model and vectorizer.
    """
    print("Starting unsupervised model training...")

    # Load dataset
    df = pd.read_csv(data_path)
    emails = df['email']
    print(f"Loaded {len(emails)} emails from dataset.")

    # Mask emails
    masked_emails = []
    start_time = time.time()
    for i, text in enumerate(emails):
        masked_text, _ = mask_pii(text)
        masked_emails.append(masked_text)
        if i % 1000 == 0 and i > 0:
            print(f"🔄 Processed {i} emails...")

    duration = time.time() - start_time
    print(f"PII masking completed in {duration:.2f} seconds.")

    # Vectorize
    print("Vectorizing masked emails...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(masked_emails)
    print(f"Vectorization complete. Shape: {X_vec.shape}")

    # Train KMeans
    print("Training KMeans model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_vec)
    print("KMeans training complete.")

    # Save model and vectorizer
    joblib.dump(kmeans, 'email_kmeans.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("KMeans model and vectorizer saved.")

    # Add cluster labels to data (for analysis)
    df['cluster'] = kmeans.predict(X_vec)
    df.to_csv('clustered_emails.csv', index=False)
    print("Cluster assignments saved to 'clustered_emails.csv'.")
    print("All done!")


if __name__ == "__main__":
    train_unsupervised_model('email_dataset.csv', n_clusters=4)
