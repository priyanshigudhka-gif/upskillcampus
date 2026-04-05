import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---- Load Model ----
model = pickle.load(open("model.pkl", "rb"))

# ---- Sidebar ----
st.sidebar.title("🌾 Project Info")
st.sidebar.write("""
**Project:** Crop Yield Prediction  
**Domain:** Agriculture  
**Model Used:** Random Forest  
**Goal:** Predict crop yield based on cost & region  
""")

# ---- Title ----
st.title("🌾 Agriculture Crop Yield Prediction")

st.write("Predict crop yield (Quintal per Hectare) using Machine Learning")

# ---- Mappings ----
crop_mapping = {
    'ARHAR': 0, 'COTTON': 1, 'GRAM': 2, 'GROUNDNUT': 3,
    'MAIZE': 4, 'MOONG': 5, 'PADDY': 6,
    'RAPESEED AND MUSTARD': 7, 'SUGARCANE': 8, 'WHEAT': 9
}

state_mapping = {
    'Uttar Pradesh': 11, 'Karnataka': 5, 'Gujarat': 3,
    'Andhra Pradesh': 1, 'Maharashtra': 6, 'Punjab': 8,
    'Haryana': 4, 'Rajasthan': 9, 'Madhya Pradesh': 7,
    'Tamil Nadu': 10, 'Bihar': 2, 'Orissa': 0, 'West Bengal': 12
}

# ---- Input Section ----
st.markdown("---")
st.subheader("🔍 Enter Input Details")

col1, col2 = st.columns(2)

with col1:
    crop_name = st.selectbox("Select Crop", list(crop_mapping.keys()))
    cost_a2fl = st.number_input(
        "Cost of Cultivation (A2+FL)",
        min_value=0.0,
        value=10000.0
    )

with col2:
    state_name = st.selectbox("Select State", list(state_mapping.keys()))
    cost_c2 = st.number_input(
        "Cost of Cultivation (C2)",
        min_value=0.0,
        value=20000.0
    )

production_cost = st.number_input(
    "Cost of Production per Quintal",
    min_value=0.0,
    value=1500.0
)

# ---- Encoding ----
crop = crop_mapping[crop_name]
state = state_mapping[state_name]

# ---- Prediction ----
st.markdown("---")

if st.button("🚀 Predict Yield"):
    input_data = np.array([[crop, state, cost_a2fl, cost_c2, production_cost]])
    prediction = model.predict(input_data)

    # Metric display (Professional UI)
    st.metric(
        label="🌱 Predicted Yield (Quintal/Hectare)",
        value=f"{prediction[0]:.2f}"
    )

# ---- Feature Importance ----
st.markdown("---")
st.subheader("📊 Feature Importance")

features = ['Crop', 'State', 'Cost_A2FL', 'Cost_C2', 'Production_Cost']
importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_title("Feature Importance")

st.pyplot(fig)

# ---- Model Info ----
st.markdown("---")

with st.expander("🧠 Model Details"):
    st.write("""
    - Model Used: Random Forest Regressor  
    - Problem Type: Regression  
    - Target Variable: Yield (Quintal per Hectare)  
    - Inputs: Crop, State, Cost of Cultivation, Cost of Production  
    """)