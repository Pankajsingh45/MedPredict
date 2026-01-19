import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, os, json

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/dataset.csv")

print("Columns in dataset:", df.columns.tolist())

target_col = "Disease"
symptom_cols = [c for c in df.columns if c != target_col]

# Fill missing with "None"
df[symptom_cols] = df[symptom_cols].fillna("None")

# Build one global list of all symptom names
unique_symptoms = set()
for col in symptom_cols:
    unique_symptoms.update(df[col].unique())
unique_symptoms = sorted(s.strip().lower().replace(" ", "_") for s in unique_symptoms if s != "None")

print(f"✅ Total unique symptoms found: {len(unique_symptoms)}")

# Create binary (multi-hot) encoded features
def encode_row(row):
    row_symptoms = [str(v).strip().lower().replace(" ", "_") for v in row.values if v != "None"]
    features = [1 if s in row_symptoms else 0 for s in unique_symptoms]
    return features

X = df[symptom_cols].apply(encode_row, axis=1, result_type='expand')
y = df[target_col]

# Encode diseases
le_disease = LabelEncoder()
y_encoded = le_disease.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_disease.classes_))

# Save artifacts
joblib.dump(model, "models/model.pkl")
joblib.dump(le_disease, "models/disease_encoder.pkl")

with open("models/symptom_list.json", "w") as f:
    json.dump(unique_symptoms, f)

with open("models/labels.json", "w") as f:
    json.dump({i: name for i, name in enumerate(le_disease.classes_)}, f)

print("\n🎯 Model, disease encoder, and symptom list saved successfully in /models.")
