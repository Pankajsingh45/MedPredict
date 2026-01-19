# app.py
import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from models import db, User, Prediction
from dotenv import load_dotenv
import joblib
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev-secret")

# Example: mysql+mysqlconnector://user:pass@localhost/medpredict
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI", "sqlite:///medpredict.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB
db.init_app(app)

# ✅ Ensure tables exist
with app.app_context():
    db.create_all()

# Load model artifacts
MODEL_DIR = "models"
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
    label_map = json.load(f)

# Helpers
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return User.query.get(uid)

@app.route("/")
def index():
    if current_user():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("User with same username/email exists", "danger")
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash("Logged in successfully", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for('login'))

# Dashboard
@app.route("/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    recent = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template("dashboard.html", user=user, recent=recent)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    import numpy as np
    import joblib, json
    from difflib import get_close_matches
    from models import db, Prediction  # ✅ add Prediction model

    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms or len(symptoms) == 0:
        return jsonify({"error": "Please enter at least one symptom"}), 400

    # Load model + encoders
    model = joblib.load("models/model.pkl")
    le_disease = joblib.load("models/disease_encoder.pkl")

    with open("models/symptom_list.json") as f:
        all_symptoms = json.load(f)

    # Create binary input vector
    input_vector = [0] * len(all_symptoms)
    recognized, unrecognized = [], []

    for s in symptoms:
        s = s.strip().lower().replace(" ", "_")
        match = get_close_matches(s, all_symptoms, n=1, cutoff=0.7)
        if match:
            idx = all_symptoms.index(match[0])
            input_vector[idx] = 1
            recognized.append(match[0])
        else:
            unrecognized.append(s)

    if sum(input_vector) == 0:
        return jsonify({"error": f"No matching symptoms found for: {', '.join(unrecognized)}"}), 400

    # Predict disease
    X_input = np.array(input_vector).reshape(1, -1)
    probs = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(probs))
    predicted_disease = le_disease.inverse_transform([pred_idx])[0]

    # Top 3 probable diseases
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [
        (le_disease.inverse_transform([i])[0], round(float(probs[i]) * 100, 2))
        for i in top3_idx
    ]

    # ✅ Save prediction to database
    user_id = session.get("user_id")
    p = Prediction(
        user_id=user_id,
        input_features=json.dumps(symptoms),
        predicted_label=predicted_disease,
        probabilities=json.dumps(top3)
    )
    db.session.add(p)
    db.session.commit()

    return jsonify({
        "recognized_symptoms": recognized,
        "unrecognized_symptoms": unrecognized,
        "predicted_disease": predicted_disease,
        "top3": top3
    })



# History page
@app.route("/history")
def history():
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    records = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", records=records)

# Optional manual DB init command
@app.cli.command("init-db")
def init_db():
    with app.app_context():
        db.create_all()
    print("✅ Database tables created successfully!")

if __name__ == "__main__":
    app.run(debug=True)
