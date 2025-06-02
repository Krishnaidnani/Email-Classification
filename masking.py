# masking.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")

REGEX_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone_number": r"\b(?:\+91[-\s]?|0)?\d{10}\b",
    "dob": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "aadhar_num": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "credit_debit_no": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "cvv_no": r"\b\d{3}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/\d{2,4}\b"
}

def mask_pii(text):
    entities = []

    # Regex matches
    for entity, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            entities.append({
                "position": [start, end],
                "classification": entity,
                "entity": match.group()
            })

    # SpaCy PERSON detection
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": ent.text
            })

    # Sort and mask
    entities = sorted(entities, key=lambda x: x["position"][0], reverse=True)
    masked_text = text
    for ent in entities:
        start, end = ent["position"]
        masked_text = masked_text[:start] + f"[{ent['classification']}]" + masked_text[end:]

    return masked_text, entities
