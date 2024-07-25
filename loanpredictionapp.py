
import streamlit as st
import pickle
import numpy as np

# Load the trained linear regression model
with open('model_lr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define mappings for categorical variables
gender_mapping = {'Male': 1, 'Female': 0}
married_mapping = {'Yes': 1, 'No': 0}
education_mapping = {'Graduate': 1, 'Not Graduate': 0}
self_employed_mapping = {'Yes': 1, 'No': 0}
property_area_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
loan_status_mapping = {'Y': 1, 'N': 0}

# Create the web app
st.title('Loan Amount Prediction App')

# User inputs
gender = st.selectbox('Gender', list(gender_mapping.keys()))
married = st.selectbox('Married', list(married_mapping.keys()))
dependents = st.number_input('Dependents', min_value=0, max_value=10, step=1)
education = st.selectbox('Education', list(education_mapping.keys()))
self_employed = st.selectbox('Self Employed', list(self_employed_mapping.keys()))
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=0)
credit_history = st.selectbox('Credit History', [1, 0])
property_area = st.selectbox('Property Area', list(property_area_mapping.keys()))
loan_status= st.selectbox('Loan_Status', list(loan_status_mapping.keys()))
loan_status_encoded = loan_status_mapping[loan_status]
# Encode user inputs
gender_encoded = gender_mapping[gender]
married_encoded = married_mapping[married]
education_encoded = education_mapping[education]
self_employed_encoded = self_employed_mapping[self_employed]
property_area_encoded = property_area_mapping[property_area]

# Create the feature array
features = np.array([[gender_encoded, married_encoded, dependents, education_encoded, self_employed_encoded,
                      applicant_income, coapplicant_income, loan_amount_term, credit_history, property_area_encoded,loan_status_encoded]], dtype=np.float64)

# Make prediction
if st.button('Predict'):
    output = model.predict(features)
    if output[0] == 1:
        st.write('Loan Approved')
    else:
        st.write('Loan Not Approved')
