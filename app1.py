import streamlit as st
import numpy as np
import joblib
from reportlab.pdfgen import canvas

# Load trained model and scaler
model = joblib.load("heart_disease_hybrid_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("💓 Heart Disease Prediction App")
st.write("Fill in the details below to check your heart disease risk.")



# Sidebar for better navigation
st.sidebar.image("heart_logo.png", use_container_width=True)
st.sidebar.write("This app helps predict heart disease based on medical parameters.")
st.sidebar.markdown("[🌐Developer Portfolio](https://nithishgowda.netlify.app/)", unsafe_allow_html=True)

# Styling
st.markdown(
    """
    <style>
        .main {background-color: #f0f2f6;}
        h1 {color: #ff4d4d;}
    </style>
    """,
    unsafe_allow_html=True,
)

# # Function to generate PDF report
# def generate_pdf(result_text):
#     pdf_file = "Heart_Disease_Report.pdf"
#     c = canvas.Canvas(pdf_file)
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(100, 750, "Heart Disease Prediction Report")
#     c.setFont("Helvetica", 12)
#     c.drawString(100, 720, result_text)
#     c.save()
#     return pdf_file

# Define Input Fields with Descriptions
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [("0 - Female", 0), ("1 - Male", 1)], format_func=lambda x: x[0])[1]
cp = st.selectbox("Chest Pain Type", [("0 - Typical Angina", 0), ("1 - Atypical Angina", 1),
                                      ("2 - Non-Anginal Pain", 2), ("3 - Asymptomatic", 3)], format_func=lambda x: x[0])[1]
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [("0 - No", 0), ("1 - Yes", 1)], format_func=lambda x: x[0])[1]
restecg = st.selectbox("Resting ECG", [("0 - Normal", 0), ("1 - ST-T Wave Abnormality", 1),
                                       ("2 - Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", [("0 - No", 0), ("1 - Yes", 1)], format_func=lambda x: x[0])[1]
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", [("0 - Upsloping", 0), ("1 - Flat", 1), ("2 - Downsloping", 2)], format_func=lambda x: x[0])[1]
ca = st.selectbox("Number of Major Vessels", [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4)], format_func=lambda x: x[0])[1]
thal = st.selectbox("Thalassemia", [("0 - Normal", 0), ("1 - Fixed Defect", 1), ("2 - Reversible Defect", 2), ("3 - Unknown", 3)], format_func=lambda x: x[0])[1]

# Prediction Function
def predict():
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)
    return prediction[0]

# Display Prediction & Generate Report
if st.button("Predict"):
    prediction = predict()
    
    if prediction == 1:
        result_text = "❌ **Heart Disease Detected!**\n\nPlease consult a doctor immediately and take necessary precautions."
        advice = "⚠️ **Advice:** Maintain a healthy diet, exercise regularly, and monitor cholesterol levels."
    else:
        result_text = "✅ **No Heart Disease Detected!**\n\nGreat job! Keep maintaining a healthy lifestyle."
        advice = "🎉 **Advice:** Continue a balanced diet, regular exercise, and routine health check-ups."
    
    st.subheader(result_text)
    st.write(advice)

    