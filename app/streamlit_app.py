import streamlit as st
import sys
import os
import locale

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import HousePricePredictor

# Set locale for currency formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, '')  # Fallback to system default

# Initialize predictor
@st.cache_resource
def load_predictor():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pickle')
    params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'params.pickle')
    return HousePricePredictor(model_path, params_path)

# Page configuration
st.set_page_config(
    page_title='Bengaluru House Price Prediction',
    page_icon='🏠',
    layout='wide'
)

st.title('🏠 Bengaluru House Price Prediction')
st.subheader('Predict the price of a house in Bengaluru using Machine Learning')

# Load the predictor
try:
    predictor = load_predictor()
    locations = predictor.get_available_locations()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        total_sq_feet = st.number_input(
            'Total Square Feet',
            min_value=300,
            max_value=100000,
            value=1200,
            help='Enter the total square feet area of the house'
        )
        
        bathrooms = st.number_input(
            'Number of Bathrooms',
            min_value=1,
            max_value=20,
            value=2,
            help='Enter the number of bathrooms in the house'
        )
    
    with col2:
        bedrooms = st.number_input(
            'Number of Bedrooms',
            min_value=1,
            max_value=20,
            value=3,
            help='Enter the number of bedrooms in the house'
        )
        
        location = st.selectbox(
            'Choose the location of the house',
            locations,
            help='Select the location in Bengaluru'
        )
    
    # Prediction button
    if st.button('Predict Price', type='primary', use_container_width=True):
        try:
            price = predictor.predict(total_sq_feet, bathrooms, bedrooms, location)
            
            if price <= 0:
                st.error('⚠️ The location you have chosen does not have any houses with the entered features')
            else:
                # Format price
                price_formatted = f"₹{price:,.2f}"
                
                # Display result in a nice card
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f8ff;
                        padding: 2rem;
                        border-radius: 10px;
                        text-align: center;
                        border: 2px solid #4CAF50;
                    ">
                        <h3 style="color: #2e7d32;">Predicted Price</h3>
                        <h1 style="color: #1b5e20; font-size: 3rem;">{price_formatted}</h1>
                        <p style="color: #555;">
                            for {bedrooms} BHK, {bathrooms} bathroom, {total_sq_feet} sqft<br>
                            at <strong>{location}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except ValueError as e:
            st.error(f'❌ Error: {str(e)}')
        except Exception as e:
            st.error(f'❌ An unexpected error occurred: {str(e)}')
    
    # Display additional info
    with st.expander('ℹ️ About this prediction'):
        st.markdown("""
        **Model Information:**
        - Model: Linear Regression
        - Features: Total Square Feet, Bathrooms, Bedrooms, Location
        - Training Data: Bengaluru House Data (13,320 records)
        
        **How it works:**
        1. Input the house features
        2. Select the location
        3. The model predicts the price based on historical data
        """)
    
    # Display available locations count
    st.sidebar.markdown("---")
    st.sidebar.info(f"📍 {len(locations)} locations available in Bengaluru")
    
    with st.sidebar.expander("📍 Sample Locations"):
        st.write(locations[:20])  # Show first 20 locations
    
except Exception as e:
    st.error(f"❌ Failed to load the prediction model: {str(e)}")
    st.info("Please make sure the model files (model.pickle and params.pickle) exist in the project root.")
