import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from data import *
import weather

# -----------------------------
# 🔹 Load or Train Model
# -----------------------------
def load_or_train_model():
    model_path = 'model/model.pkl'

    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, 'rb'))
            print("✅ Existing model loaded successfully!")
        except Exception as e:
            print("⚠️ Error loading model:", e)
            
            print("⚙️ Retraining model...")
            model = train_and_save_model(model_path)
    else:
        print("⚙️ Training new model...")
        model = train_and_save_model(model_path)

    return model


def train_and_save_model(model_path):
    # Load dataset (ensure the CSV file exists in your project folder)
    dataset_path = 'model/Crop_recommendation.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("❌ Dataset not found at 'model/Crop_recommendation.csv'")

    df = pd.read_csv(dataset_path)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    pickle.dump(model, open(model_path, 'wb'))
    print("✅ Model retrained successfully!")
    print("Features used:", list(X.columns))
    print("Target column:", y.name)
    return model


# Load the model once
model = load_or_train_model()

# -----------------------------
# 🔹 Initialize Flask App
# -----------------------------
app = Flask(__name__)


# -----------------------------
# 🔹 Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/form', methods=["GET", "POST"])
def form():
    state = request.form.get('state')
    district = request.form.get('district')

    try:
        la = weather.temp(state)
        rain = int(la[0][0])
        temp = int(la[1][0])
        humd = int(la[2][0])
    except Exception as e:
        print("⚠️ Weather data fetch error:", e)
        rain, temp, humd = 0, 0, 0

    return render_template("form.html", rain=rain, humd=humd, temp=temp)


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        N = request.form.get("N")
        P = request.form.get("P")
        K = request.form.get("K")
        temperature = request.form.get("Temp")
        humidity = request.form.get("Humd")
        PH = request.form.get("Ph")
        rainfall = request.form.get("rnfall")

        new_input = [[N, P, K, temperature, humidity, PH, rainfall]]
        new_input = np.asarray(new_input).astype(np.float32)

        new_output = model.predict(new_input)
        crop = new_output[0]

        crop_idx = {
            'rice': 'RICE', 'maize': 'MAIZE', 'chickpea': 'CHICKPEA', 'kidneybeans': 'KIDNEYBEANS',
            'pigeonpeas': 'PIGEONPEAS', 'mothbeans': 'MOTHBEANS', 'mungbean': 'MUNGBEAN',
            'blackgram': 'BLACKGRAM', 'lentil': 'LENTIL', 'pomegranate': 'POMEGRANATE',
            'banana': 'BANANA', 'mango': 'MANGO', 'grapes': 'GRAPES', 'watermelon': 'WATERMELON',
            'muskmelon': 'MUSKMELON', 'apple': 'APPLE', 'orange': 'ORANGE', 'papaya': 'PAPAYA',
            'coconut': 'COCONUT', 'cotton': 'COTTON', 'jute': 'JUTE', 'coffee': 'COFFEE'
        }

        crop = crop_idx.get(str(crop).lower(), crop.upper())

        filename = f"static/images/{crop}.png"
        crop1 = crop.lower()
        a = eval(crop1)

        return render_template(
            "output.html",
            crop=crop,
            filename=filename,
            description=a["DESCRIPTION"],
            types=a["TYPE"],
            disease=a["DISEASES"],
            companion=a["COMPANION"],
            pests=a["PESTS"],
            fertilizer=a["FERTILIZER"],
            tips=a["TIPS"],
            spacing=a["SPACING"],
            watering=a["WATERING"],
            storage=a["STORAGE"]
        )

    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template("error.html", message=str(e))


@app.route('/home')
def retry():
    return render_template('landing.html')


# -----------------------------
# 🔹 Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
