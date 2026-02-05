import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Inputs
company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight (kg)', value=1.5)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Panel', ['No', 'Yes'])
os = st.selectbox('OS', df['OpSys'].unique())
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())
inches = st.number_input('Screen Size (Inches)', value=15.6)

if st.button('Predict Price'):
    # Prepare inputs
    ts = 1 if touchscreen == 'Yes' else 0
    ips_panel = 1 if ips == 'Yes' else 0
    
    # NEW FIX: Create a DataFrame with column names matching X_train
    query_df = pd.DataFrame([[
        company, laptop_type, ram, weight, ts, ips_panel, os, cpu, inches
    ]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'OpSys', 'Cpu Brand', 'Inches'])

    # Predict
    prediction = pipe.predict(query_df)
    result = int(np.exp(prediction[0]))
    
    st.success(f"The estimated price is: ₹{result}")