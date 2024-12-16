import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Load your pre-trained model
with open('model.pkl', 'rb') as f:
    lm2 = pickle.load(f)

# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# Load the feature importance DataFrame
final_fi = load_feature_importance("feature_importance.xlsx")  

# Sidebar setup
image_sidebar = Image.open('cars1.jpeg')  
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Cars Features')

# Feature selection on sidebar
def get_user_input():
    Year = st.sidebar.number_input('Year (No)', min_value=1963, max_value=2022, step=1, value=2014)
    Engine_Size = st.sidebar.number_input('Engine Size (No)', min_value=1, max_value=9, step=1, value=3)
    Mileage = st.sidebar.number_input('Mileage (No)', min_value=0, max_value=20000000, step=10, value=100000)
    Gear_Type = st.sidebar.selectbox('Gear Type', ['Automatic', 'Manual'])
    Options = st.sidebar.selectbox('Options', ['Full', 'Semi Full', 'Standard'])
    Origin = st.sidebar.selectbox('Origin', ['Gulf Arabic', 'Saudi', 'Other', 'Unknown'])
    Region = st.sidebar.selectbox('Region', ['Region 0','Region 1', 'Region 2', 'Region 3', 'Region 4'])
    Make = st.sidebar.selectbox('Make', ['Make 0', 'Make 1', 'Make 2', 'Make 3', 'Make 4', 'Make 5'])
    Type = st.sidebar.selectbox('Type', ['Type 0', 'Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6', 
                                         'Type 7', 'Type 8'])

    user_data = {
        'Year_No': Year,
        'Engine Size_No': Engine_Size,
        'Mileage_No': Mileage,
        'Gear Type': Gear_Type,
        'Options': Options,
        'Origin': Origin,
        'Region': Region,
        'Make': Make,
        'Type': Type
    }
    return user_data

# Centered title
st.markdown("<h1 style='text-align: center;'>Saudi Arabia Used Car Price Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")
    
    # Sort feature importance DataFrame by 'Importance'
    final_fi_sorted = final_fi.sort_values(by='Importance', ascending=True)
    
    # Create interactive bar chart with Plotly
    fig = px.bar(
        final_fi_sorted,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Importance',
        color_discrete_sequence=['#48a3b4']  
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Right column: Prediction Interface
with right_col:
    st.header("Predict Used Car Price")
    
    # User inputs from sidebar
    user_data = get_user_input()

    # Transform the input into the required format
    def prepare_input(data):
        # Prepare input for model prediction
        input_data = {
            'Year_No': data['Year_No'],
            'Engine Size_No': data['Engine Size_No'],
            'Mileage_No': data['Mileage_No'],
            'Gear Type': 1 if data['Gear Type'] == 'Automatic' else 0,  # OneHot encoding
            'Options': {'Full': 1, 'Semi Full': 2, 'Standard': 3}[data['Options']],
            'Origin': {'Gulf Arabic': 1, 'Saudi': 2, 'Other': 3, 'Unknown': 4}[data['Origin']],
            'Region': {'Region 0': 0, 'Region 1': 1, 'Region 2': 2, 'Region 3': 3, 'Region 4': 4}[data['Region']],
            'Make': {'Make 0': 0, 'Make 1': 1, 'Make 2': 2, 'Make 3': 3, 'Make 4': 4, 'Make 5': 5}[data['Make']],
            'Type': {'Type 0': 0, 'Type 1': 1, 'Type 2': 2, 'Type 3': 3, 'Type 4': 4, 'Type 5': 5, 'Type 6': 6, 'Type 7': 7, 'Type 8': 8}[data['Type']]
        }
        return np.array([list(input_data.values())])

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data)
        prediction = lm2.predict(input_array)
        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")

# streamlit run UsedCar.py