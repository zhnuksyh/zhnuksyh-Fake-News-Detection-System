import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and prepare dataset
df = pd.read_csv("dataset.csv")[:4000]  # small dataset for speed
X = df["text"]
y = df["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ✅ MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=1)
mlp.fit(X_train, y_train)

# Evaluation
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"MLP Model Accuracy: {accuracy * 100:.2f}%")


# Save the model and vectorizer
joblib.dump(mlp, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
