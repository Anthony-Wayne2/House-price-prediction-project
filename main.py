import streamlit as st
import pickle
import locale

model = pickle.load(open('model.pickle', 'rb'))
params = pickle.load(open('params.pickle', 'rb'))
locale.setlocale(locale.LC_ALL, 'en_US')

st.set_page_config(page_title='Bengaluru House Price Prediction', page_icon=':house:')
st.title('Bengaluru House Price Prediction')
st.subheader('Predict the price of a house in Bengaluru using Machine Learning')

total_sq_feet = st.number_input('Total Square Feet', placeholder='Enter the total square feet area of the house',
                                min_value=300, max_value=100000)
number_bathrooms = st.number_input('Number of Bathrooms', placeholder='Enter the number of bathrooms in the house',
                                   min_value=1, max_value=20)
number_bedrooms = st.number_input('Number of Bedrooms', placeholder='Enter the number of bedrooms in the house',
                                  min_value=1, max_value=20)
location = st.selectbox('Choose the location of the house', params['columns'])

button = st.button('Predict', type='primary')

if button:
    features = [0 for i in range(len(params['columns']) + params['prefix'])]
    features[0] = total_sq_feet
    features[1] = number_bathrooms
    features[2] = number_bedrooms
    features[params['prefix'] + params['columns'].index(location)] = 1
    price = model.predict([features])[0] * 100000
    if price <= 0:
        st.error('The location you have chosen does not have any houses with the entered features')
    else:
        price = locale.currency(price, grouping=True)
        st.success(f'The predicted price of the house is {price}')
