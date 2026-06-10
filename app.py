import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Credit Risk Engine", page_icon="🏦", layout="wide")

# Dashboard Title
st.title("Enterprise Credit Risk Engine 🏦")
st.markdown("Predict customer credit default risk in real-time using our Machine Learning Pipeline.")

# Load the saved model (Pipeline)
@st.cache_resource
def load_model():
    # Yahan apne .pkl file ka sahi path dalein
    return joblib.load('/content/industrial_credit_risk_pipeline.pkl')

try:
    pipeline = load_model()
    st.sidebar.success("Model Loaded Successfully! ✅")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    pipeline = None

st.markdown("### Enter Customer Details")

# Creating input fields in 3 columns for a clean UI
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
    annual_income = st.number_input("Annual Income ($)", value=75000.0)
    loan_amount = st.number_input("Loan Amount ($)", value=15000.0)
    credit_score = st.number_input("Credit Score", value=710.0)
    transaction_count = st.number_input("Transaction Count", value=30.0)
    spending_ratio = st.number_input("Spending Ratio", value=0.25)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Education", "Business", "Other"])
    repayment_history = st.selectbox("Repayment History", ["Good", "Fair", "Poor"])

with col3:
    join_year = st.number_input("Join Year", min_value=2000, max_value=2030, value=2021)
    join_month = st.number_input("Join Month (1-12)", min_value=1, max_value=12, value=5)
    customer_tenure_years = st.number_input("Customer Tenure (Years)", value=3.0)
    age_group = st.selectbox("Age Group", ["Young", "Adult", "Senior"])
    loan_to_income_ratio = st.number_input("Loan to Income Ratio", value=0.20)

st.markdown("---")

# Predict Button
if st.button("Evaluate Credit Risk 🚀", type="primary"):
    if pipeline is not None:
        # Pura data ek dictionary mein pack karke DataFrame banayein
        input_dict = {
            'age': age, 'gender': gender, 'region': region,
            'education_level': education_level, 'employment_type': employment_type,
            'annual_income': annual_income, 'loan_amount': loan_amount,
            'loan_purpose': loan_purpose, 'credit_score': credit_score,
            'repayment_history': repayment_history, 'transaction_count': transaction_count,
            'spending_ratio': spending_ratio, 'join_year': join_year,
            'join_month': join_month, 'customer_tenure_years': customer_tenure_years,
            'age_group': age_group, 'loan_to_income_ratio': loan_to_income_ratio
        }
        
        input_df = pd.DataFrame([input_dict])
        
        try:
            # Prediction aur Probability nikalna
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1] # Default (1) ki probability
            
            # Results display karna
            st.subheader("Risk Assessment Result:")
            if prediction == 1:
                st.error(f"🚨 HIGH RISK DETECTED")
                st.write(f"**Probability of Default:** {probability:.2%}")
                st.write("**Action:** 🛑 Decline Application")
            else:
                st.success(f"✅ LOW RISK (APPROVED)")
                st.write(f"**Probability of Default:** {probability:.2%}")
                st.write("**Action:** 🟢 Approve Application")
                
        except Exception as e:
            st.error(f"Prediction mein error aayi: {str(e)}")
    else:
        st.error("Model load nahi hua hai. Kripya file path check karein.")