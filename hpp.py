import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------------------
# Load trained model + mappings
# ------------------------------
@st.cache_resource
def load_model():
    with open("housepricemodel.pkl", "rb") as f:
        model = pickle.load(f)
    return model

with open("city_encoder.pkl", "rb") as f:
    city_encoder = pickle.load(f)

with open("city_zip_mapping.pkl", "rb") as f:
    city_zip_mapping = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = load_model()

st.title("🏡 House Price Prediction App")
st.write("Enter details below to predict house price:")

# ------------------------------
# Numeric inputs
# ------------------------------

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=9, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=6, value=2)
sqft_living = st.number_input("Living Area (sqft)", min_value=200, max_value=15000, value=1500, step=1)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=200, max_value=10000000, value=4000, step=1)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
waterfront = st.number_input("Waterfront", min_value=0, max_value=9)
view = st.number_input("View (0–4)", min_value=0, max_value=4, value=0)
condition = st.number_input("Condition (1–5)", min_value=1, max_value=5, value=3)
sqft_above = st.number_input("Above Ground Area (sqft)", min_value=200, max_value=20000, value=1500, step=1)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=10000, value=0, step=1)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

# Renovation input
renovated = st.radio("Has the house been renovated?", ["No", "Yes"])
if renovated == "Yes":
    yr_renovated = st.number_input("Year Renovated", min_value=yr_built, max_value=2025, value=2020)
else:
    yr_renovated = 0

# ------------------------------
# Prepare features
# ------------------------------
# Select city + zip
city_str = st.selectbox("City", list(city_zip_mapping.keys()))
zip_code = st.selectbox("Zip Code", city_zip_mapping[city_str])

# Encode city properly
city_encoded = city_encoder.transform([city_str])[0]
sqft_lot_transformed = np.log1p(sqft_lot)

input_df = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot_transformed,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "city": city_encoded,
    "zip": zip_code
}])
input_scaled = scaler.transform(input_df)
# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"🏠 Estimated Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
