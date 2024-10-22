import streamlit as st
import pandas as pd
import pickle

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
 image_path = "C:/Users/rajij/Downloads/singapore image.jpg" 
 st.image(image_path, use_column_width=1000)

elif select=='Modelling':

   st.title('Predictive Flat Resale Price')
   

   st.write("Input the details below:")
   town_d = ['ANG MO KIO','BEDOK','BISHAN','BUKIT BATOK','BUKIT MERAH','BUKIT PANJANG','CHOA CHU KANG','CLEMENTI','GEYLANG','HOUGANG','JURONG EAST','JURONG WEST','KALLANG/WHAMPOA','Others','PASIR RIS','PUNGGOL','QUEENSTOWN','SEMBAWANG','SENGKANG','SERANGOON','TAMPINES','TOA PAYOH','WOODLANDS','YISHUN']
   flat_type_d = ['3 ROOM', '4 ROOM', '5 ROOM', 'Others']
   flat_model_d = ['Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2','New Generation', 'Others', 'Premium Apartment', 'Simplified', 'Standard']
   block_d=['309', '216', '211', '202', '235', '232', '308', '220', '219',
       '247', '320', '252', '223', '230', '329', '313', '117', '110',
       '343', '345', '346', '121', '129', '130', '128', '127', '126',
       '403', '404', '405', '417', '418', '419', 'Others', '435', '424',
       '425', '534', '611', '505', '503', '610', '607', '524', '513',
       '643', '542', '550', '330', '333', '156', '152', '209', '231',
       '254', '103', '105', '324', '120', '124', '414', '427', '428',
       '150', '336', '401', '716', '620', '101', '18', '28', '29', '30',
       '501', '502', '504', '2', '20', '21', '59', '58', '55', '22',
       '104', '107', '33', '46', '116', '115', '125', '138', '412', '402',
       '416', '136', '510', '218', '213', '532', '533', '536', '537',
       '44', '540', '712', '50', '54', '15', '34', '36', '35', '42', '53',
       '51', '8', '166', '82', '134', '132', '131', '133', '62', '422',
       '507', '508', '43', '707', '609', '165', '710', '613', '602',
       '605', '112', '1', '31', '111', '118', '137', '139', '24', '304',
       '310', '160', '161', '164', '205', '248', '4', '6', '146', '143',
       '145', '228', '227', '244', '113', '163', '169', '204', '135',
       '225', '123', '307', '221', '214', '142', '140', '141', '7', '40',
       '77', '119', '114', '5', '12', '16', '13', '108', '102', '106',
       '63', '60', '3', '19', '26', '37', '47', '109', '57', '66', '264',
       '9', '10', '633', '440', '509', '334', '323', '322', '326', '314',
       '312', '354', '506', '305', '306', '303', '339', '520', '703',
       '704', '706', '411', '311', '301', '415', '413', '201', '45', '65',
       '23', '38', '17', '319', '327', '410', '239', '154', '341', '251',
       '302', '234', '237', '233', '242', '243', '257', '210', '208',
       '316', '408', '240', '331', '406', '167', '168', '170', '172',
       '185', '186', '187', '212', '215', '516', '511', '535', '217',
       '554', '203', '407', '11', '14', '52', '32', '171', '173', '25',
       '267', '250', '226', '236', '420', '157', '272', '253', '159',
       '245', '149', '148', '27', '206', '701', '229', '238', '328',
       '512', '523', '155', '612', '122', '426', '626', '709', '207',
       '249', '409', '518', '317', '321', '325', '241', '528', '514',
       '64', '153', '151', '147', '222', '421', '631', '714', '144',
       '351', '246', '521', '263', '158', '541', '332', '423', '627']
   street_name_d = ['ANG MO KIO AVE 1','ANG MO KIO AVE 10','ANG MO KIO AVE 3','ANG MO KIO AVE 4','ANG MO KIO AVE 5','BEDOK NTH RD','BEDOK NTH ST 3','BEDOK RESERVOIR RD','BT BATOK WEST AVE 6','CIRCUIT RD','CLEMENTI AVE 4','HOUGANG AVE 8','JURONG EAST ST 21','JURONG WEST ST 42','LOR 1 TOA PAYOH','MARSILING DR','Others','SIMEI ST 1','TAMPINES ST 21','TAMPINES ST 22','YISHUN RING RD']
   
   floor_area_sq_ft_d = [301,312,333,365,376,398,409,419,430,441,452,462,473,484,495,505,516,517,527,538,548,559,570, 581,592,602,607,613,624,635,636,637,645,649,656,667,678,679,688,696,697,698,699,710,721,723,
                           731,734,740,742,744,750,752,753,764,775,785,786,796,805,806,807,816,818,828,839,850,861,871,
                           882,893,894,904,914,925,936,937,947,948,957,959,968,977,979,990,1001,1011,1022,1033,1044,
                           1054,1065,1076,1078,1087,1097,1108,1119,1130,1140,1151,1162,1173,1184,1194,1205,1216,1227,
                           1237,1248,1259,1270,1280,1291,1302,1313,1323,1334,1345,1356,1367,1377,1388,1399,1410,1411,
                           1420,1431,1442,1453,1463,1474,1485,1496,1506,1517,1528,1539,1550,1560,1571,1582,1593,1603,
                           1614,1625,1636,1640,1646,1657,1668,1679,1689,1700,1711,1722,1732,1743,1754,1765,1776,1786,
                           1797,1808,1819,1829,1840,1851,1862,1872,1883,1894,1905,1915,1926,1937,1948,1959,1969,1980,
                           1991,2002,2012,2023,2034,2038,2045,2066,2077,2098,2131,2142,2217,2228,2238,2260,2314,2378,
                           2389,2421,2551,2572,2594,2615,2647,2680,2690,2787,2809,2863,3013,3196,3304,3947]
                              
   floor_median_d= [2,3,5,8,11,13,14,17,18,20,23,26,28,29,32,33,35,38,41,44,47,50]
    
   with st.form(key='input_form'):
      col1, col2 = st.columns([1,1])
      with col1:
         town_s= st.selectbox('town', town_d, index=None, key=1)
         block_s = st.selectbox('block', block_d, index=None, key=4)
         flat_type_s = st.selectbox('flat_type', flat_type_d, index=None, key=2)
         flat_model_s = st.selectbox('flat_model', flat_model_d, index=None, key=3)
         stname_s = st.selectbox('Street Name', street_name_d, index=None, key=5)


      with col2: 
         remaining_lease_s = st.number_input("remaining_lease, min_value=49, max_value=99", min_value=49, max_value=99)
         floor_area_sq_ft_s = st.selectbox('floor_area_sq_ft', floor_area_sq_ft_d, index=None, key=7)
         age_of_property_s = st.number_input("age_of_property, min_value=0, max_value=58", min_value=0, max_value=58)
         Floor_median_s = st.selectbox('Floor_median', floor_median_d, index=None, key=8)

      submit_button = st.form_submit_button(label='Submit')   

      if  submit_button:

         if not age_of_property_s or floor_area_sq_ft_s is None or age_of_property_s is None or Floor_median_s is None or block_s is None or town_s is None or flat_type_s is None or stname_s is None:
            st.error('Please fill in all the above fields!')
         else:
            
            input_data = {
              
               'town': [town_s],
               'street_name': [stname_s],
               'flat_type': [flat_type_s],
               'block': [block_s],
               'flat_model': [flat_model_s],
               'remaining_lease': [remaining_lease_s],
               'floor_area_sq_ft': [floor_area_sq_ft_s],
               'age_of_property': [age_of_property_s],
               'Floor_median': [Floor_median_s]       
            }

            
            input_data_ = pd.DataFrame(input_data)

            st.table(input_data_)

            with open('dcr.pkl', 'rb') as file:
               dcr = pickle.load(file)

            with open('C:/Users/rajij/Streamlit_Home Page/Singapore flat resale model/OE.pkl', 'rb') as file2:
               loaded_Encoder = pickle.load(file2)

            with open('OE1.pkl', 'rb') as file2:
               loaded_Encoder1 = pickle.load(file2)

            with open('OE2.pkl', 'rb') as file2:
               loaded_Encoder2 = pickle.load(file2)

            with open('OE3.pkl', 'rb') as file2:
               loaded_Encoder3 = pickle.load(file2)

            with open('OE4.pkl', 'rb') as file2:
               loaded_Encoder4 = pickle.load(file2)


            with open('scaler.pkl', 'rb') as file3:
               loaded_scaler = pickle.load(file3)





            

            input_data_['town'] = loaded_Encoder.transform(input_data_[['town']])
            input_data_['flat_type'] = loaded_Encoder1.transform(input_data_[['flat_type']])
            input_data_['block'] = loaded_Encoder2.transform(input_data_[['block']])
            input_data_['street_name'] = loaded_Encoder3.transform(input_data_[['street_name']])
            input_data_['flat_model'] = loaded_Encoder4.transform(input_data_[['flat_model']])

            input_data_['floor_area_sq_ft'] = np.log(input_data_['floor_area_sq_ft'])

            input_data_['Floor_median'] = np.log(input_data_['Floor_median'])

            input_data_['remaining_lease'] = np.log(input_data_['remaining_lease'])
            input_data_['age_of_property'] = np.log(input_data_['age_of_property'])
   
            input_data_=input_data_[['town','flat_type', 'block', 'flat_model', 'remaining_lease', 'floor_area_sq_ft', 'age_of_property', 'Floor_median','street_name']]
            
            input_data_s = loaded_scaler.transform(input_data_)
            
            st.dataframe(input_data_)

            new_pred = dcr.predict(input_data_s)[0]

            
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))




