
import joblib
import pandas as pd


MODEL_PATH = "models/stroke_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_stroke(patient_data):
    model = load_model()

    data = pd.DataFrame([patient_data])

    probability = model.predict_proba(data)[0][1]

    prediction = int(probability >= 0.5)

    return prediction, probability