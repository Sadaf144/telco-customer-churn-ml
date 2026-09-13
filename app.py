%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .banner-img {
        border-radius: 12px;
        margin-bottom: 25px;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'churn_model.pkl')
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['scaler'], artifacts['label_encoders']

try:
    model, scaler, label_encoders = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header Layout with Image
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.title("⚡ Enterprise Customer Churn Platform")
    st.markdown("### AI-Powered Predictive Analytics & Retention Automation")
    st.caption("Identify churn risks early, analyze customer sentiment patterns, and export bulk enterprise data reports in real time.")

with header_col2:
    # High quality banner image from Unsplash
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=600&q=80",
        caption="Predictive Analytics Dashboard",
        use_column_width=True
    )

st.markdown("---")

tab1, tab2 = st.tabs(["👤 Single Customer Diagnostic", "📁 Enterprise Batch Analytics"])

# ==================== TAB 1: INDIVIDUAL DIAGNOSTIC ====================
with tab1:
    st.markdown("### 📋 Customer Profile Input")
    
    with st.expander("👤 Personal & Account Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with col3:
            tenure = st.slider("Tenure Length (Months)", min_value=0, max_value=72, value=12)

    with st.expander("🌐 Services & Subscriptions", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with st.expander("💳 Contract & Billing Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col2:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Run Risk Diagnostic", type="primary", use_container_width=True):
        raw_inputs = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([raw_inputs])

        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        st.markdown("---")
        st.subheader("🎯 Risk Diagnostic Results")
        
        m_col1, m_col2 = st.columns([1, 1])

        with m_col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={'suffix': "%"},
                title={'text': "Calculated Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if prediction == 1 else "#00c853"},
                    'steps': [
                        {'range': [0, 40], 'color': "#e8f5e9"},
                        {'range': [40, 70], 'color': "#fffde7"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ]
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with m_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error("⚠️ Status: HIGH CHURN RISK")
                st.write("**Recommended Action:** Target immediately with a retention offer, contract upgrade incentive, or personalized discount.")
            else:
                st.success("✅ Status: LOW CHURN RISK")
                st.write("**Recommended Action:** Account is in good health. Target for cross-selling premium options.")

# ==================== TAB 2: BATCH CSV PROCESSING ====================
with tab2:
    st.subheader("📁 Batch Dataset Processing")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        
        st.markdown("### Uploaded Data Preview")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("⚡ Execute Bulk Churn Analysis", type="primary"):
            df_proc = raw_df.copy()

            if 'customerID' in df_proc.columns:
                df_proc = df_proc.drop('customerID', axis=1)

            if 'TotalCharges' in df_proc.columns:
                df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'].astype(str).str.strip(), errors='coerce')
                df_proc['TotalCharges'].fillna(df_proc['TotalCharges'].median(), inplace=True)

            if 'Churn' in df_proc.columns:
                df_proc = df_proc.drop('Churn', axis=1)

            for col, le in label_encoders.items():
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else 0
                    )

            scaled_features = scaler.transform(df_proc)
            predictions = model.predict(scaled_features)
            probabilities = model.predict_proba(scaled_features)[:, 1] * 100

            results_df = raw_df.copy()
            results_df['Churn Probability (%)'] = probabilities.round(2)
            results_df['Risk Category'] = np.where(predictions == 1, 'High Risk', 'Low Risk')

            st.success("Batch Prediction Completed!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers Evaluated", f"{len(results_df):,}")
            c2.metric("At-Risk Customers", f"{int((predictions == 1).sum()):,}")
            c3.metric("Overall Churn Exposure", f"{((predictions == 1).sum() / len(results_df)) * 100:.1f}%")

            st.markdown("### 📊 Risk Distribution")
            fig_hist = px.histogram(
                results_df, 
                x="Churn Probability (%)", 
                color="Risk Category",
                color_discrete_map={"High Risk": "#ff4b4b", "Low Risk": "#00c853"},
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Prediction Results Table")
            st.dataframe(results_df, use_container_width=True)

            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Risk Analysis Report (CSV)",
                data=csv_data,
                file_name='churn_predictions_report.csv',
                mime='text/csv'
            )
