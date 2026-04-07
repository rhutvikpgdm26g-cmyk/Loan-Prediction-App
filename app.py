import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.title("💳 Loan Approval Prediction")
st.write("Select a model and enter details")

# Load encoders
encoders = joblib.load("label_encoders.pkl")

# Load all models
models = {
    "Decision Tree": joblib.load("DecisionTree_model.pkl"),
    "Bagging": joblib.load("Bagging_model.pkl"),
    "AdaBoost": joblib.load("AdaBoost_model.pkl"),
    "Random Forest": joblib.load("Randomforest_model.pkl"),
    "Random Forest (Tuned)": joblib.load("Randomforest_Hypertuned_model.pkl")
}

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# --- INPUTS ---
st.subheader("Enter Applicant Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])

# Button
if st.button("Check Eligibility"):

    # Encoding
    gender_val = encoders['Gender'].transform([gender])[0]
    married_val = encoders['Married'].transform([married])[0]
    education_val = encoders['Education'].transform([education])[0]
    self_employed_val = encoders['Self_Employed'].transform([self_employed])[0]
    property_area_val = encoders['Property_Area'].transform([property_area])[0]

    dependents_val = 3 if dependents == "3+" else int(dependents)
    credit_history_val = int(credit_history)

    # Feature array
    input_data = np.array([[
        gender_val,
        married_val,
        dependents_val,
        education_val,
        self_employed_val,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history_val,
        property_area_val
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 'Y':
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # --- CHART ---
    st.subheader("📊 Financial Overview")

    labels = ["Applicant Income", "Loan Amount"]
    values = [applicant_income, loan_amount]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    st.pyplot(fig)

    # --- PROBABILITY ---
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]

        st.subheader("📈 Prediction Confidence")
        st.write(f"Approval Probability: {prob[1]*100:.2f}%")