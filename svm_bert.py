import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
df = pd.read_csv("updated_file.csv")  # Ensure this file has processed queries, labels, and numeric features
print("1")

# Identify numeric features (excluding 'query' and 'label')
numeric_features = df.drop(columns=["payload", "label"]).values  # Convert to NumPy array
print("2")

# Step 1️⃣: Load Pre-trained BERT for Feature Extraction
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("3")
# Function to generate BERT embeddings
def get_bert_embeddings(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract [CLS] token representation 
print("4")

# Step 2️⃣: Convert Queries into Numerical Features
#bert_features = np.array([get_bert_embeddings(query) for query in df["payload"]])  # BERT embeddings

bert_features = []
for i, query in enumerate(df["payload"]):
    try:
        print(f"Processing query {i+1}/{len(df)}")
        emb = get_bert_embeddings(query)
        bert_features.append(emb)
    except Exception as e:
        print(f"Error at query {i+1}: {str(e)}")
        bert_features.append(np.zeros(768))  # Add a zero vector for failed cases

bert_features = np.array(bert_features)

X = np.hstack((bert_features, numeric_features))  # Combine BERT features with numeric features
y = df["label"].values  # Target labels (0 for benign, 1 for malicious)
print("5")

# Step 3️⃣: Train SVM on Extracted Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

svm_model = SVC(kernel="linear", C=1.0)  # Linear kernel for classification
svm_model.fit(X_train, y_train)

# Step 4️⃣: Evaluate Model Performance
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

import joblib

# Save the trained SVM model
joblib.dump(svm_model, "svm_bert_model.pkl")
print("✅ Model saved as svm_bert_model.pkl")