import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("C:\\Users\\chaud\Downloads\\archive (1).zip")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Create pipeline: scaling + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline Model Saved Successfully!")