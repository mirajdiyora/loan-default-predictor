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

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Gradient Headers & Titles */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Containers */
    .stCard {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Result Box Styling */
    .result-box-approved {
        background: linear-gradient(135deg, #064E3B 0%, #047857 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .result-box-denied {
        background: linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border-left: 4px solid #6366F1;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
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
        st.error("❌ Model file `model.pkl` not found! Please run `python train_model.py` first.")
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

# Helper function to preprocess single dictionary into 24 features
def encode_inputs(raw_dict, feature_names):
    # Initialize zero dictionary for all 24 features
    row = {feat: 0 for feat in feature_names}
    
    # Numerical features directly assigned
    num_feats = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
    for num in num_feats:
        if num in raw_dict:
            row[num] = float(raw_dict[num])
            
    # Categorical One-Hot Encoding mapping matching drop_first=True
    # Education
    edu = raw_dict.get("Education")
    if edu == "High School":
        row["Education_High School"] = 1
    elif edu == "Master's":
        row["Education_Master's"] = 1
    elif edu == "PhD":
        row["Education_PhD"] = 1
        
    # EmploymentType
    emp = raw_dict.get("EmploymentType")
    if emp == "Part-time":
        row["EmploymentType_Part-time"] = 1
    elif emp == "Self-employed":
        row["EmploymentType_Self-employed"] = 1
    elif emp == "Unemployed":
        row["EmploymentType_Unemployed"] = 1
        
    # MaritalStatus
    mar = raw_dict.get("MaritalStatus")
    if mar == "Married":
        row["MaritalStatus_Married"] = 1
    elif mar == "Single":
        row["MaritalStatus_Single"] = 1
        
    # HasMortgage
    if raw_dict.get("HasMortgage") == "Yes":
        row["HasMortgage_Yes"] = 1
        
    # HasDependents
    if raw_dict.get("HasDependents") == "Yes":
        row["HasDependents_Yes"] = 1
        
    # LoanPurpose
    purp = raw_dict.get("LoanPurpose")
    if purp == "Business":
        row["LoanPurpose_Business"] = 1
    elif purp == "Education":
        row["LoanPurpose_Education"] = 1
    elif purp == "Home":
        row["LoanPurpose_Home"] = 1
    elif purp == "Other":
        row["LoanPurpose_Other"] = 1
        
    # HasCoSigner
    if raw_dict.get("HasCoSigner") == "Yes":
        row["HasCoSigner_Yes"] = 1
        
    df_single = pd.DataFrame([row], columns=feature_names)
    return df_single

# ==========================================
# PAGE 1: SINGLE APPLICANT PREDICTOR
# ==========================================
if nav_option == "🎯 Single Applicant Predictor":
    st.markdown("<h1 class='main-title'>Loan Default Risk Assessor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Enter applicant financial and personal details to generate real-time loan default probability and credit decision recommendations.</p>", unsafe_allow_html=True)
    
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
                    <h1 style='font-size: 3.5rem; margin: 0;'>{repay_prob:.1f}%</h1>
                    <p style='font-size: 1.1rem; opacity: 0.9;'>Probability of On-Time Loan Repayment</p>
                    <hr style='border-color: rgba(255,255,255,0.2);'>
                    <p>Estimated Default Risk Score: <strong>{default_prob:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box-denied'>
                    <h2>⚠️ HIGH DEFAULT RISK - CAUTION / DENIAL ADVISED</h2>
                    <h1 style='font-size: 3.5rem; margin: 0;'>{default_prob:.1f}%</h1>
                    <p style='font-size: 1.1rem; opacity: 0.9;'>Probability of Loan Default</p>
                    <hr style='border-color: rgba(255,255,255,0.2);'>
                    <p>Repayment Probability: <strong>{repay_prob:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("#### 🔍 Key Risk Factors Analysis")
            
            # Key indicator highlights
            risk_flags = []
            if dti_ratio > 0.5:
                risk_flags.append(f"🔴 High Debt-To-Income Ratio ({dti_ratio:.2f})")
            if credit_score < 580:
                risk_flags.append(f"🔴 Poor Credit Score ({credit_score})")
            if employment_type == "Unemployed":
                risk_flags.append("🔴 Applicant is currently Unemployed")
            if interest_rate > 18.0:
                risk_flags.append(f"🔴 High Interest Rate burden ({interest_rate:.1f}%)")
            if loan_amount > income * 2:
                risk_flags.append(f"🔴 High Loan-to-Income ratio (${loan_amount:,.0f} vs ${income:,.0f})")
                
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
            
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Applicants", len(out_df))
            c2.metric("Low Risk (Approved)", sum(out_df["Risk_Assessment"] == "Low Risk (Approved)"))
            c3.metric("High Risk (Default)", sum(out_df["Risk_Assessment"] == "High Risk (Default)"))
            
            st.dataframe(out_df, use_container_width=True)
            
            # Download CSV
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
    st.markdown("<p class='sub-title'>Detailed breakdown of model evaluation metrics, accuracy, recall, and feature importance.</p>", unsafe_allow_html=True)
    
    # Metrics Row
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
        st.subheader("🌲 Feature Importance Breakdown")
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(12)
            
            st.bar_chart(fi_df.set_index("Feature"))
        else:
            st.info("Feature importance chart is available for Tree-based models.")
            
    with col_b:
        st.subheader("ℹ️ Dataset Information")
        st.json({
            "Dataset": "Loan_default.csv",
            "Total Records": "255,347",
            "Training Split": "204,277 (80%)",
            "Test Split": "51,070 (20%)",
            "Target Class": "Default (1) vs Non-Default (0)",
            "Class Distribution": "Default: 11.6%, Non-Default: 88.4%"
        })

# ==========================================
# PAGE 4: DEPLOYMENT GUIDE & LINK
# ==========================================
elif nav_option == "🚀 Deployment Guide & Link":
    st.markdown("<h1 class='main-title'>Deploying Your App & Web Link</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Follow these simple steps to deploy this app online and get a public live URL for free!</p>", unsafe_allow_html=True)
    
    st.success("🎉 All required files (`model.pkl`, `app.py`, `requirements.txt`, `.streamlit/config.toml`) have been generated!")
    
    st.markdown("""
    ### 🌐 Option 1: Streamlit Community Cloud (Recommended - 100% Free & Fast)
    
    1. **Upload Code to GitHub**:
       - Create a new repository on GitHub (e.g. `loan-default-streamlit`).
       - Push all files from this folder (`app.py`, `model.pkl`, `feature_names.json`, `metadata.json`, `requirements.txt`, `.streamlit/`).
       
    2. **Deploy on Streamlit Cloud**:
       - Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
       - Click **"New app"**.
       - Select your repository: `loan-default-streamlit`
       - Main file path: `app.py`
       - Click **"Deploy!"**
       
    3. **Your Live Link**:
       - Once deployed, Streamlit gives you a public shareable URL like:
         `https://your-app-name.streamlit.app`
    
    ---
    
    ### 🤖 Option 2: Hugging Face Spaces (Free Alternative)
    
    1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
    2. Select **Streamlit** as the Space SDK.
    3. Upload `app.py`, `model.pkl`, `requirements.txt`, and related json files.
    4. Your app will build automatically and yield a public link like:
       `https://huggingface.co/spaces/yourusername/loan-default-predictor`
    """)

