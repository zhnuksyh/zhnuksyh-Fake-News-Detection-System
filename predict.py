import pandas as pd
import joblib

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load a dataset — you can load 'dataset.csv' or any other file
df = pd.read_csv('dataset.csv')  # this contains 'text' and 'label'

# OPTIONAL: Just take a small chunk if you want lightweight
df = df.sample(30, random_state=42)  # randomly choose 10 samples for prediction

# Transform the text using the vectorizer
X = vectorizer.transform(df['text'])

# Predict using the model
predictions = model.predict(X)

# Add the predictions into the dataframe
df['prediction'] = predictions

# Display results
for idx, row in df.iterrows():
    print(f"\n NEWS: {row['text'][:100]}...")  # show only first 100 characters
    print(f"    ACTUAL: {row['label']}")
    print(f"    PREDICTED: {row['prediction']}")
