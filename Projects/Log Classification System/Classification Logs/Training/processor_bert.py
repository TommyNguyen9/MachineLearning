from sentence_transformers import SentenceTransformer
import joblib

import os
print(os.getcwd())

transformer_model = None

def get_model():
    global transformer_model
    if transformer_model is None:
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return transformer_model

classifier_model = joblib.load(r"C:\Users\Smoke Nandos\Desktop\Computer Science Uni\Machine Learning\Projects\Log Classification System\Classification Logs\Models\log_classifier.joblib")


def classify_with_bert(log_message):
    model = get_model()
    
    message_embedding = transformer_model.encode(log_message)
    probabilities = classifier_model.predict_proba([message_embedding])[0]
   
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = classifier_model.predict([message_embedding])[0]

    return predicted_label


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error!",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE 404 len: 1583 time: 0. 1878400",
        "System crashed due to drives error when restarting server",
        "YO WHATS UP?",
        "Multiple login failures occured on user 9393 account",
        "Server A340 restarted unexpectedly during the process of data transfer!"
    ]

    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)


    

