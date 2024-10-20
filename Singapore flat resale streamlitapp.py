import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout='wide')

with st.sidebar:
    select= option_menu("Main Page",["HomePage","Modelling"])

if select =="HomePage":
 st.title("Singapore  Resale Flat Prices Predicting Model")
 st.subheader("Singapore  Resale Flat Prices Predicting Model using Pythonscripting, Data Preprocessing, EDA, and Streamlit ")


elif select=='Modelling':

   st.title('Predictive Flat Resale Price')
 # User input for features
   st.write("Input the details below:")
   town_d = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
   flat_type_d = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
   flat_model_d = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified','Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace','2-room', 'Improved-Maisonette', 'Multi Generation','Premium Apartment', 'Adjoined flat', 'Premium Maisonette', 'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft', '3Gen'] 

   with st.form(key='input_form'):
      col1, col2 = st.columns([1,1])
      with col1:
         town_s= st.selectbox('town', town_d, index=None, key=1)
         block_s = st.text_input("block")
         flat_type_s = st.selectbox('flat_type', flat_type_d, index=None, key=2)
         flat_model_s = st.selectbox('flat_model', flat_model_d, index=None, key=3)


      with col2: 
         lease_commence_date_s = st.number_input("lease_commence_date", min_value=1900, max_value=2024)
         remaining_lease_s = st.number_input("remaining_lease", min_value=0, max_value=99)
         floor_area_sq_ft_s = st.number_input("floor_area_sq_ft", min_value=0.00, max_value=100000000.00)
         age_of_property_s = st.number_input("age_of_property", min_value=0, max_value=500)
         Floor_median_s = st.number_input("Floor", min_value=0, max_value=200)

      submit_button = st.form_submit_button(label='Predict')   


      if  submit_button:

         if not lease_commence_date_s or not remaining_lease_s or not floor_area_sq_ft_s  or not age_of_property_s or not Floor_median_s or town_s is None or not block_s  or flat_type_s is None:
            st.error('Please fill in all the above fields!')
         else:
            
            input_data = {
              
               'town': [town_s],
               'flat_type': [flat_type_s],
               'block': [block_s],
               'flat_model': [flat_model_s],
               'lease_commence_date': [lease_commence_date_s],
               'remaining_lease': [remaining_lease_s],
               
               'floor_area_sq_ft': [floor_area_sq_ft_s],
               'age_of_property': [age_of_property_s],
               'Floor_median': [Floor_median_s]       
            }

            
            input_data_ = pd.DataFrame(input_data)

            st.table(input_data_)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/dcr.pkl', 'rb') as file:
               dcr = pickle.load(file)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/OE.pkl', 'rb') as file2:
               loaded_Encoder = pickle.load(file2)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/OE1.pkl', 'rb') as file2:
               loaded_Encoder1 = pickle.load(file2)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/OE2.pkl', 'rb') as file2:
               loaded_Encoder2 = pickle.load(file2)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/OE4.pkl', 'rb') as file2:
               loaded_Encoder4 = pickle.load(file2)


            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/scaler.pkl', 'rb') as file3:
               loaded_scaler = pickle.load(file3)





            

            input_data_['town'] = loaded_Encoder.transform(input_data_[['town']])
            input_data_['flat_type'] = loaded_Encoder1.transform(input_data_[['flat_type']])
            input_data_['block'] = loaded_Encoder2.transform(input_data_[['block']])
            input_data_['flat_model'] = loaded_Encoder4.transform(input_data_[['flat_model']])

            input_data_['lease_commence_date'] = np.log(input_data_['lease_commence_date'])

            input_data_['Floor_median'] = np.log(input_data_['Floor_median'])
   
            input_data_=input_data_[[ 'town','flat_type', 'block', 'flat_model','lease_commence_date', 'remaining_lease', 'floor_area_sq_ft', 'age_of_property', 'Floor_median']]
            
            input_data_s = loaded_scaler.transform(input_data_)
            st.dataframe(input_data_)

            new_pred = dcr.predict(input_data_s)[0]
            
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))




