import streamlit as st
import pandas as pd
import joblib
import os

DEPLOY_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_model.pkl")

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

st.title("🧳 Wellness Tourism Package — Purchase Predictor")
st.write("Enter customer details to predict whether they are likely to purchase the Wellness Tourism Package.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips (avg/year)", min_value=0, max_value=20, value=2)

with col2:
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000, step=1000)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)

# Build a raw input dataframe — NO manual encoding needed.
# The pipeline's ColumnTransformer (StandardScaler + OneHotEncoder) handles it internally.
input_data = {
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch,
}

input_df = pd.DataFrame([input_data])

st.divider()

if st.button("Predict", type="primary"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ This customer is **likely to purchase** the Wellness Package.")
    else:
        st.warning("❌ This customer is **unlikely to purchase** the Wellness Package.")

    st.metric("Purchase Probability", f"{probability:.1%}")

    with st.expander("See input data sent to the model"):
        st.dataframe(input_df)
