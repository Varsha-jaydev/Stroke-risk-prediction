import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/stroke_model.pkl"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #0f172a;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .risk-high {
        background-color: #fee2e2;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #dc2626;
    }

    .risk-low {
        background-color: #dcfce7;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #16a34a;
    }

    .disclaimer {
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🧠 Stroke Risk Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Machine learning application for estimating stroke probability
    from demographic and health-related features.
    </div>
    """,
    unsafe_allow_html=True,
)


if model is None:

    st.error(
        "Model not found. Please train the model first with:"
    )

    st.code("python train.py")

    st.stop()


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=50.0,
        step=1.0,
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
    )

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"],
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        ["No", "Yes"],
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["Yes", "No"],
    )


with col2:

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ],
    )

    residence_type = st.selectbox(
        "Residence Type",
        ["Urban", "Rural"],
    )

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        max_value=400.0,
        value=100.0,
        step=0.1,
    )

    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown",
        ],
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict Stroke Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    patient = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    patient_df = pd.DataFrame([patient])

    probability = model.predict_proba(patient_df)[0][1]

    prediction = probability >= 0.5

    st.subheader("Prediction")

    probability_percent = probability * 100

    if prediction:

        st.markdown(
            f"""
            <div class="risk-high">

            <h2>⚠️ Higher Predicted Risk</h2>

            <h3>{probability_percent:.2f}% predicted probability</h3>

            The model classified this input as having a higher
            predicted probability of stroke.

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">

            <h2>✅ Lower Predicted Risk</h2>

            <h3>{probability_percent:.2f}% predicted probability</h3>

            The model classified this input as having a lower
            predicted probability of stroke.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(
        min(probability, 1.0),
        text=f"Predicted probability: {probability_percent:.2f}%",
    )


# ---------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------

st.markdown(
    """
    <div class="disclaimer">

    ⚠️ <strong>Disclaimer:</strong>
    This application is an educational machine-learning project.
    It is not a medical diagnostic tool and should not be used
    to make healthcare decisions.

    </div>
    """,
    unsafe_allow_html=True,
)
