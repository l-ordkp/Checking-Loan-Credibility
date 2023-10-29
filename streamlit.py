import streamlit as st
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load the pre-trained model
model = joblib.load("rf_model.joblib")  

# Define a function to predict loan approval
def predict_loan_approval(data):
    prediction = model.predict(data)
    return prediction

# Create a Streamlit web app
st.title("Loan Approval Prediction")

st.write("Fill in the following information to predict loan approval:")

# Collect user inputs
gender = st.radio("Gender", ["Male", "Female"])
married = st.radio("Marital Status", ["Married", "Single"])
education = st.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.radio("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term (in months)")
credit_history = st.radio("Credit History", ["1.0", "0.0"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Preprocess the input data as needed
# For example, encode categorical variables and scale numerical variables

# Prepare the input data as a list or array
input_data = [gender, married, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, property_area]

# Predict loan approval
if st.button("Predict Loan Approval"):
    # Perform any necessary data preprocessing here

    # Convert categorical data to numerical using encoding

    # Create a feature vector from user inputs
    feature_vector = [float(input_data[i]) for i in range(4)]  # The first four inputs are numeric

    # Perform the prediction
    prediction = predict_loan_approval([feature_vector])

    # Display the prediction
    if prediction[0] == 1:
        st.success(" Your loan may get approved.")
    else:
        st.error(" Your loan may not get approved.")


