# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 03:29:58 2022

@author: Ayush Gautam
"""

# importing the dependencies
import numpy as np
import pickle
import streamlit as st

# loading the model from the saved file trained_model.sav
loaded_model = pickle.load(open('C:/Users/Ayush Gautam/Documents/Work Life/FK/Higher Studies/Projects/SVM/Diabetes Prediction Model/trained_model.sav', 'rb'))

# defining a function for diabetes prediction
def diabetes_prediction(input_data):
    # input data to the classifier/model

    # converting the tuple in array using numpy
    input_arr = np.asarray(input_data)

    # reshaping the array for predicting in classifier/model
    input_arr_reshape = input_arr.reshape(1, -1)

    # predicting the input data
    prediction = loaded_model.predict(input_arr_reshape)

    # output depending upon prediction
    if prediction[0] == 0:
        return "No diabetes"
    else:
        return "You have diabetes"
    pass

def main():
    # giving title to our web app
    st.title("Diabetes Prediction Web Application")
    
    # getting input data from user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")    
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin value")
    BMI = st.text_input("Body Mass Index (BMI)")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of Patient")
    
    # string for storing the result
    report = ""
    
    # button
    if st.button("Diabetes Prediction"):
        report = diabetes_prediction((Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age))
    
    st.success(report)
    
    
if __name__ == '__main__':
    main()
