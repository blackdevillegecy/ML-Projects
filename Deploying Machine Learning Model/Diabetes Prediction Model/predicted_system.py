# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Ayush Gautam
"""

# importing the dependencies
import numpy as np
import pickle

# loading the model from the saved file trained_model.sav
loaded_model = pickle.load(open('C:/Users/Ayush Gautam/Documents/Work Life/FK/Higher Studies/Projects/SVM/Diabetes Prediction Model/trained_model.sav', 'rb'))

# input data to the classifier/model
input_data = (10, 115, 0, 0, 0, 35.3, 0.134, 29)

# converting the tuple in array using numpy
input_arr = np.asarray(input_data)

# reshaping the array for predicting in classifier/model
input_arr_reshape = input_arr.reshape(1, -1)

# predicting the input data
prediction = loaded_model.predict(input_arr_reshape)

# output depending upon prediction
if prediction[0] == 0:
    print("No diabetes")
else:
    print("You have diabetes")