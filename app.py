import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings


warnings.filterwarnings('ignore')

with open('models/scaler.pkl','rb') as s:
    scaler=joblib.load(s)


with open('models/best_linear_regression_model.pkl','rb')as m:
    model=joblib.load(m) 

st.set_page_config(page_title="Algerian Forest Fire Prediction",page_icon="🔥")
st.title("🔥Algerian Forest Fire Prediction")

columns1,columns2=st.columns(2)

with columns1:
    TEMPERATURE=st.number_input("Temperature",max_value=42,min_value=22,value=32)
    RH=st.number_input("Relative Humidity (RH)",max_value=90,min_value=21,value=62)
    WS=st.number_input("Wind Speed",max_value=29,min_value=6,value=15)
    RAIN=st.number_input("Rain",max_value=17,min_value=0,value=1,)
    FFMC=st.number_input("FFMC",max_value=96,min_value=28,value=77)
with columns2:
    DMC=st.number_input("DMC",max_value=65,min_value=0,value=14)
    ISI=st.number_input("ISI",value=4,min_value=0,max_value=19)
    CLASSES=st.selectbox("Select an Option",options=[0,1])
    REGION=st.selectbox("Select an option",options=[0,1])
    


if st.button("Predict",type='primary'):
    input_data=np.array([[TEMPERATURE,RH,WS,RAIN,FFMC,DMC,ISI,CLASSES,REGION]])
    scaled=scaler.transform(input_data)
    result=model.predict(scaled)
    st.success(result[0])