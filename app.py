import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import warnings


warnings.filterwarnings('ignore')

scaler=StandardScaler()


with open('models/best_linear_regression_model.pkl','rb')as f:
    model=joblib.load(f)

st.set_page_config(page_title="Algerian Forest Fire Prediction",page_icon="🔥")
st.title("🔥Algerian Forest Fire Prediction")

columns1,columns2=st.columns(2)

with columns1:
    TEMPERATURE=st.number_input("Temperature")
    RH=st.number_input("Relative Humidity (RH)")
    WS=st.number_input("Wind Speed")
    RAIN=st.number_input("Rain")
    FFMC=st.number_input("FFMC")
with columns2:
    DMC=st.number_input("DMC")
    ISI=st.number_input("ISI")
    CLASSES=st.number_input("Classes")
    REGION=st.number_input("Region")

input_data=np.array([[TEMPERATURE,RH,WS,RAIN,FFMC,DMC,ISI,CLASSES,REGION]])
scaled=scaler.fit_transform(input_data)
result=model.predict(scaled)

if st.button("Predict",type='primary'):
    st.success(result[0])