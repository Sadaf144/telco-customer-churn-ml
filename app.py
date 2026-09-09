import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Intelligence Platform", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    with open('churn_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['scaler'], artifacts['label_encoders']

model, scaler, label_encoders = load_artifacts()

st.title("📊 Customer Churn Intelligence Platform")
st.write("Analyze individual customer risk or upload a batch CSV dataset for instant enterprise reporting.")

# Mode Selector Tab
tab1, tab2 = st.tabs(["📁 Batch CSV Prediction (Bulk)", "👤 Individual Prediction (Single)"])

# ==================== TAB 1: BATCH CSV UPLOAD ====================
with tab1:
    st.subheader("Upload Customer Dataset (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Raw Uploaded Data Preview:")
        st.dataframe(raw_df.head())

        if st.button("Run Bulk Churn Prediction", type="primary"):
            df_proc = raw_df.copy()

            # Drop customerID if present
            customer_ids = df_proc['customerID'] if 'customerID' in df_proc.columns else None
            if 'customerID' in df_proc.columns:
                df_proc = df_proc.drop('customerID', axis=1)

            # Cleaning TotalCharges
            if 'TotalCharges' in df_proc.columns:
                df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'].astype(str).str.strip(), errors='coerce')
                df_proc['TotalCharges'].fillna(df_proc['TotalCharges'].median(), inplace=True)

            # Drop Churn target column if included in input file
            if 'Churn' in df_proc.columns:
                df_proc = df_proc.drop('Churn', axis=1)

            # Encoding Categorical Features
            for col, le in label_encoders.items():
                if col in df_proc.columns:
                    # Handle unseen categories smoothly
                    df_proc[col] = df_proc[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else 0
                    )

            # Scaling Features
            scaled_features = scaler.transform(df_proc)

            # Predictions
            predictions = model.predict(scaled_features)
            probabilities = model.predict_proba(scaled_features)[:, 1] * 100

            # Attach Results back to original DataFrame
            results_df = raw_df.copy()
            results_df['Churn_Probability (%)'] = probabilities.round(2)
            results_df['Risk_Status'] = np.where(predictions == 1, 'High Risk', 'Low Risk')

            st.success("Analysis Completed Successfully!")

            # Summary Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers Analyzed", len(results_df))
            m2.metric("High Risk Customers", (predictions == 1).sum())
            m3.metric("Overall Churn Rate (%)", f"{((predictions == 1).sum() / len(results_df)) * 100:.1f}%")

            st.write("### Prediction Results Preview:")
            st.dataframe(results_df[['customerID', 'Churn_Probability (%)', 'Risk_Status'] if customer_ids is not None else results_df.columns])

            # CSV Download Button
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Risk Analysis Report (CSV)",
                data=csv_data,
                file_name='churn_predictions_report.csv',
                mime='text/csv'
            )

# ==================== TAB 2: INDIVIDUAL PREDICTION ====================
with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

    with col2:
        st.subheader("Subscription Details")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("Billing & Contract")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=tenure * monthly_charges)

    if st.button("Analyze Single Customer Risk", type="primary", use_container_width=True):
        raw_inputs = {
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([raw_inputs])

        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        st.subheader("Result")
        r1, r2 = st.columns(2)
        r1.metric(label="Churn Probability", value=f"{probability:.1f}%")
        if prediction == 1:
            r2.error("⚠️ Status: High Risk")
        else:
            r2.success("✅ Status: Low Risk")
