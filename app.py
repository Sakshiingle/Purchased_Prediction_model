import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set up page configuration
st.set_page_config(
    page_title="Target Prediction Dashboard",
    page_icon="🎯",
    layout="centered"
)

# Load the fitted GaussianNB model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    
    # App Header
    st.title("🎯 Customer Prediction Dashboard")
    st.markdown("""
    Provide customer demographic metrics in the sidebar panel to see real-time classification predictions.
    """)
    st.write("---")

    # Sidebar Layout for Feature Inputs
    st.sidebar.header("👤 Customer Demographics")
    st.sidebar.markdown("Adjust characteristics to run a prediction.")

    # Feature 1: Gender
    gender_input = st.sidebar.selectbox("Gender", options=["Female", "Male"])
    # Convert text to numeric encoding (0 for Female, 1 for Male)
    # Note: Modify this mapping if your model was trained on a different encoding
    gender_encoded = 0 if gender_input == "Female" else 1

    # Feature 2: Age
    age_input = st.sidebar.slider("Age (Years)", min_value=18, max_value=100, value=35, step=1)

    # Feature 3: Estimated Salary
    salary_input = st.sidebar.number_input("Estimated Annual Salary ($)", min_value=1000, max_value=500000, value=50000, step=1000)

    # Reconstruct the feature matrix matching exact training names
    input_df = pd.DataFrame([{
        'Gender': gender_encoded,
        'Age': age_input,
        'EstimatedSalary': salary_input
    }])

    # Main Panel - Prediction Outcome Display
    st.subheader("📊 Classification Result")

    if st.button("Analyze Input Profile", type="primary"):
        with st.spinner("Processing demographics..."):
            # Execute Model Prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display Outcome Cards
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Class Output", value=f"Class {prediction}")
            with col2:
                confidence = prediction_proba[prediction] * 100
                st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
                
            # Quick visual indicator bar
            st.progress(float(prediction_proba[prediction]))

except FileNotFoundError:
    st.error("🚨 **File Error:** Could not locate `model.pkl` in the repository root folder.")
except Exception as e:
    st.error(f"🚨 **Runtime Encountered An Issue:** {e}")
