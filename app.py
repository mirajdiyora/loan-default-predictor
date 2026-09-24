import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Global CSS Overrides for Bulletproof Aesthetics & Contrast on local + Streamlit Cloud
st.markdown("""
<style>
    /* Force main app background & base text color */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    
    /* Main Content Headers */
    .main-title {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #818CF8, #C084FC, #F472B6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.3rem !important;
    }
    .sub-title {
        color: #CBD5E1 !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }

    /* Force all headings (h1, h2, h3, h4) to be crisp white/indigo */
    h1, h2, h3, h4, h5, h6, .stMarkdown h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    
    /* Force all widget labels (Age, Income, Education Level, etc.) to be highly visible */
    label, .stWidgetLabel, div[data-testid="stWidgetLabel"] p, label p {
        color: #E2E8F0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Input fields (Number input, Selectbox, Text input) styling */
    input, select, textarea, div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown popover list items */
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    
    /* Form Container */
    div[data-testid="stForm"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 1.8rem !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Form Submit Button & Action Buttons */
    div[data-testid="stFormSubmitButton"] button, button[kind="primary"] {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* SIDEBAR STYLING FIX - Force Dark background and Crisp White Text */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        color: #F8FAFC !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* Result Outcome Cards */
    .result-box-approved {
        background: linear-gradient(135deg, #064E3B 0%, #047857 100%) !important;
        border: 1px solid #10B981 !important;
        border-radius: 14px !important;
        padding: 1.8rem !important;
        color: #FFFFFF !important;
        text-align: center !important;
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3) !important;
    }
    .result-box-denied {
        background: linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%) !important;
        border: 1px solid #EF4444 !important;
        border-radius: 14px !important;
        padding: 1.8rem !important;
        color: #FFFFFF !important;
        text-align: center !important;
        box-shadow: 0 10px 20px rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B !important;
        border-left: 4px solid #6366F1 !important;
        padding: 1.2rem !important;
        border-radius: 10px !important;
        margin-bottom: 1rem !important;
        border: 1px solid #334155 !important;
    }
    .metric-val {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }
    .metric-lbl {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
    }
    
    /* Dataframes & Tables */
    div[data-testid="stDataFrame"] {
        background-color: #1E293B !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache model and metadata loading
@st.cache_resource
def load_assets():
    model_path = "model.pkl"
    feature_path = "feature_names.json"
    metadata_path = "metadata.json"
    
    if not os.path.exists(model_path):
        st.error("❌ Model file `model.pkl` not found! Please run `python train_level50.py` first.")
        st.stop()
        
    model = joblib.load(model_path)
    
    with open(feature_path, "r") as f:
        feature_names = json.load(f)
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    return model, feature_names, metadata

model, feature_names, metadata = load_assets()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-building.png", width=70)
    st.title("Loan Risk AI Portal")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation",
        [
            "🎯 Single Applicant Predictor",
            "📁 Batch CSV Predictor",
            "📊 Model Insights & Performance",
            "🚀 Deployment Guide & Link"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Quick Specs")
    st.caption(f"**Model:** {metadata['metrics']['Model_Type']}")
    st.caption(f"**Recall:** {metadata['metrics']['Recall'] * 100:.1f}%")
    st.caption(f"**ROC-AUC:** {metadata['metrics']['ROC_AUC']:.3f}")

# Helper function to preprocess single dictionary into high correlation features
def encode_inputs(raw_dict, feature_names):
    # Base dictionary
    base = {}
    
    # 1. Base numericals
    age = float(raw_dict.get("Age", 42))
    income = float(raw_dict.get("Income", 75000))
    loan_amount = float(raw_dict.get("LoanAmount", 50000))
    credit_score = float(raw_dict.get("CreditScore", 680))
    months_emp = float(raw_dict.get("MonthsEmployed", 60))
    num_credit = float(raw_dict.get("NumCreditLines", 3))
    interest_rate = float(raw_dict.get("InterestRate", 10.5))
    loan_term = float(raw_dict.get("LoanTerm", 36))
    dti_ratio = float(raw_dict.get("DTIRatio", 0.35))
    
    base["Age"] = age
    base["Income"] = income
    base["LoanAmount"] = loan_amount
    base["CreditScore"] = credit_score
    base["MonthsEmployed"] = months_emp
    base["NumCreditLines"] = num_credit
    base["InterestRate"] = interest_rate
    base["LoanTerm"] = loan_term
    base["DTIRatio"] = dti_ratio
    
    # 2. Advanced Engineered Ratios (Top Risk Correlation Factors)
    monthly_income = income / 12.0
    est_monthly_pay = (loan_amount * (1 + (interest_rate / 100))) / loan_term
    
    base["Loan_To_Income"] = loan_amount / (income + 1)
    base["Monthly_Income"] = monthly_income
    base["Estimated_Monthly_Payment"] = est_monthly_pay
    base["Payment_To_Income_Ratio"] = est_monthly_pay / (monthly_income + 1)
    base["Employment_Stability_Ratio"] = months_emp / (age * 12 + 1)
    base["Credit_Risk_Score"] = (850 - credit_score) * dti_ratio * (1 + interest_rate / 100)

    # 3. Categorical one-hot mapping matching train_level50.py
    edu = raw_dict.get("Education")
    base["Education_High School"] = 1 if edu == "High School" else 0
    base["Education_Master's"] = 1 if edu == "Master's" else 0
    base["Education_PhD"] = 1 if edu == "PhD" else 0
    
    emp = raw_dict.get("EmploymentType")
    base["EmploymentType_Part-time"] = 1 if emp == "Part-time" else 0
    base["EmploymentType_Self-employed"] = 1 if emp == "Self-employed" else 0
    base["EmploymentType_Unemployed"] = 1 if emp == "Unemployed" else 0
    
    mar = raw_dict.get("MaritalStatus")
    base["MaritalStatus_Married"] = 1 if mar == "Married" else 0
    base["MaritalStatus_Single"] = 1 if mar == "Single" else 0
    
    base["HasMortgage_Yes"] = 1 if raw_dict.get("HasMortgage") == "Yes" else 0
    base["HasDependents_Yes"] = 1 if raw_dict.get("HasDependents") == "Yes" else 0
    
    purp = raw_dict.get("LoanPurpose")
    base["LoanPurpose_Business"] = 1 if purp == "Business" else 0
    base["LoanPurpose_Education"] = 1 if purp == "Education" else 0
    base["LoanPurpose_Home"] = 1 if purp == "Home" else 0
    base["LoanPurpose_Other"] = 1 if purp == "Other" else 0
    
    base["HasCoSigner_Yes"] = 1 if raw_dict.get("HasCoSigner") == "Yes" else 0

    # Build row DataFrame matching selected feature_names order
    row = {feat: base.get(feat, 0) for feat in feature_names}
    df_single = pd.DataFrame([row], columns=feature_names)
    return df_single

# ==========================================
# PAGE 1: SINGLE APPLICANT PREDICTOR
# ==========================================
if nav_option == "🎯 Single Applicant Predictor":
    st.markdown("<h1 class='main-title'>Loan Default Risk Assessor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Enter applicant financial and personal details to generate real-time loan default probability powered by Level-50 Risk Correlation AI.</p>", unsafe_allow_html=True)
    
    with st.form("loan_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Applicant Profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=42, step=1)
            education = st.selectbox("Education Level", ["Bachelor's", "High School", "Master's", "PhD"])
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            months_employed = st.number_input("Months Employed", min_value=0, max_value=300, value=60, step=1)
            marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
            has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            
        with col2:
            st.subheader("💰 Financial Profile")
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=75000, step=1000)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=680, step=5)
            dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.01, max_value=0.99, value=0.35, step=0.01, help="Total monthly debt payments divided by gross monthly income.")
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=1, max_value=20, value=3, step=1)
            has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
            has_cosigner = st.selectbox("Has Co-Signer?", ["No", "Yes"])

        with col3:
            st.subheader("📋 Loan Request Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=50000, step=1000)
            interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=35.0, value=10.5, step=0.1)
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
            loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
            
            st.write("")
            st.write("")
            submit_btn = st.form_submit_button("⚡ Predict Loan Default Risk", use_container_width=True)

    if submit_btn:
        raw_data = {
            "Age": age, "Education": education, "EmploymentType": employment_type,
            "MonthsEmployed": months_employed, "MaritalStatus": marital_status,
            "HasDependents": has_dependents, "Income": income, "CreditScore": credit_score,
            "DTIRatio": dti_ratio, "NumCreditLines": num_credit_lines, "HasMortgage": has_mortgage,
            "HasCoSigner": has_cosigner, "LoanAmount": loan_amount, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "LoanPurpose": loan_purpose
        }
        
        encoded_df = encode_inputs(raw_data, feature_names)
        
        # Predict class & probability
        pred_class = model.predict(encoded_df)[0]
        pred_proba = model.predict_proba(encoded_df)[0]
        
        default_prob = pred_proba[1] * 100
        repay_prob = pred_proba[0] * 100
        
        st.markdown("---")
        st.subheader("📊 Assessment Summary & Risk Report")
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            if pred_class == 0:
                st.markdown(f"""
                <div class='result-box-approved'>
                    <h2>✅ LOW DEFAULT RISK - RECOMMENDED FOR APPROVAL</h2>
                    <h1 style='font-size: 3.5rem; margin: 0; color: #FFFFFF !important;'>{repay_prob:.1f}%</h1>
                    <p style='font-size: 1.1rem; opacity: 0.9;'>Probability of On-Time Loan Repayment</p>
                    <hr style='border-color: rgba(255,255,255,0.2);'>
                    <p>Estimated Default Risk Score: <strong>{default_prob:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box-denied'>
                    <h2>⚠️ HIGH DEFAULT RISK - CAUTION / DENIAL ADVISED</h2>
                    <h1 style='font-size: 3.5rem; margin: 0; color: #FFFFFF !important;'>{default_prob:.1f}%</h1>
                    <p style='font-size: 1.1rem; opacity: 0.9;'>Probability of Loan Default</p>
                    <hr style='border-color: rgba(255,255,255,0.2);'>
                    <p>Repayment Probability: <strong>{repay_prob:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("#### 🔍 Top Risk Drivers Identified")
            
            risk_flags = []
            lti_ratio = loan_amount / (income + 1)
            if lti_ratio > 0.8:
                risk_flags.append(f"🔴 High Loan-to-Income Ratio ({lti_ratio:.2f})")
            if dti_ratio > 0.5:
                risk_flags.append(f"🔴 High Debt-To-Income Ratio ({dti_ratio:.2f})")
            if credit_score < 580:
                risk_flags.append(f"🔴 Poor Credit Score ({credit_score})")
            if employment_type == "Unemployed":
                risk_flags.append("🔴 Applicant is currently Unemployed")
            if interest_rate > 18.0:
                risk_flags.append(f"🔴 High Interest Rate burden ({interest_rate:.1f}%)")
                
            if risk_flags:
                for flag in risk_flags:
                    st.write(flag)
            else:
                st.write("🟢 No major high-risk indicators detected!")
                
            st.markdown("#### 💡 Underwriting Advice")
            if pred_class == 0:
                st.info("Applicant demonstrates healthy financial ratios, stable employment, or adequate credit score. Standard loan approval workflow applies.")
            else:
                st.warning("High default risk detected. Consider requesting a co-signer, reducing loan amount, or requiring collateral before approval.")

# ==========================================
# PAGE 2: BATCH CSV PREDICTOR
# ==========================================
elif nav_option == "📁 Batch CSV Predictor":
    st.markdown("<h1 class='main-title'>Batch Loan Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload a CSV file containing applicant records or test with sample records from the dataset.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Applicant CSV File", type=["csv"])
    with col2:
        st.write("Or test with dataset samples:")
        load_sample = st.button("📥 Load 10 Sample Records from Dataset", use_container_width=True)
        
    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample:
        if os.path.exists("Loan_default.csv"):
            full_df = pd.read_csv("Loan_default.csv")
            batch_df = full_df.sample(10, random_state=42).reset_index(drop=True)
        else:
            st.error("Loan_default.csv dataset file not found.")

    if batch_df is not None:
        st.subheader("📋 Input Applicant Records")
        st.dataframe(batch_df.head(10), use_container_width=True)
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            results_list = []
            probs_list = []
            
            for idx, row in batch_df.iterrows():
                row_dict = row.to_dict()
                enc = encode_inputs(row_dict, feature_names)
                pred = model.predict(enc)[0]
                prob = model.predict_proba(enc)[0][1]
                
                results_list.append("High Risk (Default)" if pred == 1 else "Low Risk (Approved)")
                probs_list.append(round(prob * 100, 2))
                
            out_df = batch_df.copy()
            out_df["Risk_Assessment"] = results_list
            out_df["Default_Probability_%"] = probs_list
            
            st.markdown("---")
            st.subheader("✅ Batch Prediction Results")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Applicants", len(out_df))
            c2.metric("Low Risk (Approved)", sum(out_df["Risk_Assessment"] == "Low Risk (Approved)"))
            c3.metric("High Risk (Default)", sum(out_df["Risk_Assessment"] == "High Risk (Default)"))
            
            st.dataframe(out_df, use_container_width=True)
            
            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="loan_predictions_result.csv",
                mime="text/csv"
            )

# ==========================================
# PAGE 3: MODEL INSIGHTS & PERFORMANCE
# ==========================================
elif nav_option == "📊 Model Insights & Performance":
    st.markdown("<h1 class='main-title'>Model Analytics & Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Detailed breakdown of Level-50 model evaluation metrics, accuracy, recall, and top correlated features.</p>", unsafe_allow_html=True)
    
    m = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-lbl'>Accuracy</div>
            <div class='metric-val'>{m['Accuracy']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-lbl'>Recall (Sensitivity)</div>
            <div class='metric-val'>{m['Recall']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-lbl'>Precision</div>
            <div class='metric-val'>{m['Precision']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-lbl'>ROC-AUC Score</div>
            <div class='metric-val'>{m['ROC_AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔥 Top 10 High Correlation Risk Factors")
        if "top_correlations" in metadata:
            corr_df = pd.DataFrame(
                list(metadata["top_correlations"].items()),
                columns=["Feature / Driver", "Absolute Correlation"]
            ).sort_values("Absolute Correlation", ascending=True)
            st.bar_chart(corr_df.set_index("Feature / Driver"))
        else:
            st.info("Correlation analysis data loaded.")
            
    with col_b:
        st.subheader("ℹ️ Dataset Information")
        st.json({
            "Dataset": "Loan_default.csv",
            "Total Records": "255,347",
            "Training Split": "204,277 (80%)",
            "Test Split": "51,070 (20%)",
            "Model Architecture": "HistGradientBoosting + Feature Engineering",
            "Class Balance": "Default: 11.6%, Non-Default: 88.4%"
        })

# ==========================================
# PAGE 4: DEPLOYMENT GUIDE & LINK
# ==========================================
elif nav_option == "🚀 Deployment Guide & Link":
    st.markdown("<h1 class='main-title'>Deploying Your App & Web Link</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Follow these simple steps to deploy this app online and get a public live URL for free!</p>", unsafe_allow_html=True)
    
    st.success("🎉 Level-50 Model & files (`model.pkl`, `feature_names.json`, `metadata.json`, `app.py`) are ready!")
    
    st.markdown("""
    ### 🌐 Streamlit Community Cloud (100% Free & Fast)
    
    1. **Upload Code to GitHub**:
       - Go to your repository on GitHub.
       - Upload the updated `model.pkl`, `feature_names.json`, `metadata.json`, and `app.py`.
       
    2. **Automatic Deployment**:
       - Streamlit Cloud will automatically detect the changes and update your live website in ~30 seconds!
    """)

