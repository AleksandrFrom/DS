import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('fraud_detection_pipline.pkl')

st.title('Fraud Detection Predict App')

st.markdown('Enter transaction details and use the predict button')

st.divider()

transaction_type = st.selectbox('Transaction Type', ['PAYMENT','TRANSFER', 'CASH_OUT', 'DEPOSIT'])

amount = st.number_imput('Amount', min_value = 0.0, value = 1000.0)
oldbalanceOrg = st.number_input('Old balance (Sender)', min_value = 0.0, value = 100000)
newbalanceOrig = st.number_input('New Balance (Sender)', min_value = 0.0, value = 9000)
oldbalanceDest = st.number_input('Old Balance (Recevier)', min_value= 0.0, value = 0.0)
newbalanceDest = st.number_input('New Balance (Recevier)', min_value= 0.0, value = 0.0)

if st.button('Predist'):
    input_data = pd.DataFrame([{
        'type' : transaction_type,
        'amount' : amount,
        'oldbalanceOrg' : oldbalanceOrg,
        'newbalanceOrig' : newbalanceOrig,
        'oldbalanceDest' : oldbalanceDest,
        'newbalanceDest' : newbalanceDest
    }])

    prediction = model.predict(input_data)

    st.subheader(f"Prediction : '{int(prediction)}' ()")

    if prediction == 1:
        st.error('This transaction can be fraud')
    else:
        st.success('This transaction looks like it is not a fraud')