import streamlit as st
import joblib
import pandas as pd
import pickle
import numpy as np
# Load the trained model
with open('telcocustomerchurn.pkl', 'rb') as file:
    model = pickle.load(file)
   
y_pred = model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]])
print(y_pred)

# Streamlit app
st.title('Telco Customer Churn Prediction')

# Input fields
customer_id = st.text_input('Customer ID')
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 1)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=500.0)
churn = st.selectbox('Churn', ['Yes', 'No'])

# Convert inputs to a dataframe
input_data = {
    'CustomerID': [customer_id],
    'gender': [1 if gender == 'Male' else 0],
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'Partner': [1 if partner == 'Yes' else 0],
    'Dependents': [1 if dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == 'Yes' else 0],
    'MultipleLines': [1 if multiple_lines == 'Yes' else 0 if multiple_lines == 'No' else 2],
    'InternetService': [0 if internet_service == 'DSL' else 1 if internet_service == 'Fiber optic' else 2],
    'OnlineSecurity': [1 if online_security == 'Yes' else 0 if online_security == 'No' else 2],
    'OnlineBackup': [1 if online_backup == 'Yes' else 0 if online_backup == 'No' else 2],
    'DeviceProtection': [1 if device_protection == 'Yes' else 0 if device_protection == 'No' else 2],
    'TechSupport': [1 if tech_support == 'Yes' else 0 if tech_support == 'No' else 2],
    'StreamingTV': [1 if streaming_tv == 'Yes' else 0 if streaming_tv == 'No' else 2],
    'StreamingMovies': [1 if streaming_movies == 'Yes' else 0 if streaming_movies == 'No' else 2],
    'Contract': [0 if contract == 'Month-to-month' else 1 if contract == 'One year' else 2],
    'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
    'PaymentMethod': [0 if payment_method == 'Electronic check' else 1 if payment_method == 'Mailed check' else 2 if payment_method == 'Bank transfer (automatic)' else 3],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Churn': [1 if churn == 'Yes' else 0],
}

input_df = pd.DataFrame(input_data)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_df.drop(columns=['CustomerID']))
    if prediction[0] == 1:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is likely to stay.')
else:
        st.success('The customer is likely to stay.')