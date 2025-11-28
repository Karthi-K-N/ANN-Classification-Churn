import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the trained model, encoder, scaler
model = load_model('model.h5')

with open('onehot_endcode_geo.pkl', 'rb') as file:
    label_encode_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encode_gender=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# user input
geography=st.selectbox('Geography', label_encode_geo.categories_[0])
gender=st.selectbox('Gender', label_encode_gender.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#onehot encoded Geography
geo_encoded=label_encode_geo.transform([[geography]]).toarray()
geo_encode_df=pd.DataFrame(geo_encoded, columns=label_encode_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True), geo_encode_df], axis=1)

input_scaled_data=scaler.transform(input_data)

prediction=model.predict(input_scaled_data)
prediction_prob=prediction[0][0]

if prediction_prob > 0.5:
    st.write(f"Churn probability: {prediction_prob:.2f} - customer is likely to churn")
else:
    st.write(f"Churn probability: {prediction_prob:.2f} - customer is not likely to churn" )
