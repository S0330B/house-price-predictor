import streamlit as st
import joblib
import pandas as pd
import numpy as np

pipe = joblib.load('voting_regressor_model.joblib')
st.title('House Price Predictor')
area = st.number_input('Area in square feet',min_value=1)
bedrooms = st.number_input('Number of bedrooms',min_value=1)
bathrooms = st.number_input('Number of bathrooms',min_value=1)
stories = st.number_input('Number of stories',min_value=1)
parking = st.number_input("Number of parking avalable",min_value=1)
mainroad = st.selectbox("Is the house on the main road? ", ['Yes', 'No'])
airconditioning = st.selectbox("Does the house have air conditioning ?", ['Yes', 'No'])
prefarea = st.selectbox("Is the house in the preffered area ?",['Yes','No'])
furnishingstatus = st.selectbox("What is the furnishing status?",["Furnished","Semi-Furnished","Unfurnished"])

if st.button("Predict Price"):
    try:
        bathrooms_per_area = bathrooms / area if area != 0 else 0
        bedrooms_per_area = bedrooms / area if area !=0 else 0
        bath_bed_ratio = bathrooms / (bedrooms + 1)
        rooms_total = bedrooms + bathrooms + parking

        mainroad_bin = 1 if mainroad == 'Yes' else 0
        airconditioning_bin = 1 if airconditioning == 'Yes' else 0
        prefarea_bin = 1 if prefarea == 'Yes' else 0

        query_df = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad_bin,
            'airconditioning': airconditioning_bin,
            'prefarea': prefarea_bin,
            'bathrooms_per_area': bathrooms_per_area,
            'bedrooms_per_area': bedrooms_per_area,
            'bath_bed_ratio': bath_bed_ratio,
            'rooms_total': rooms_total,
            'furnishingstatus': furnishingstatus
        }])

        predicted_price_log = pipe.predict(query_df)[0]
        predicted_price = np.expm1(predicted_price_log)
        st.success(f"The predicted price is: {predicted_price:,.0f}")
        
    except Exception as e:
        st.error(f"⚠️ Error: {e}")




