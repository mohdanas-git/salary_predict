import streamlit as st
import pandas as pd
import numpy as np

st.header("Salary Prediction App")
st.markdown("Enter your years of experience to predict your potential salary.")
name = st.text_input("Enter Your Name: ")
exp = st.selectbox("Enter Your Experience: ",(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30))
button = st.button("Calculate")
data = pd.read_csv("Salary_Data.csv")
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
a = exp
features = np.array([[a]])
sal = int(model.predict(features))
if button:
    st.subheader(f"""
    Hello, {name} you have {exp} year of experience so your salary expect to be {sal} Rs
    """)