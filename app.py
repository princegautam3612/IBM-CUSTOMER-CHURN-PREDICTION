import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

pipeline = joblib.load("churn_pipeline.pkl")
model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocessor']

st.title("📉 Customer Churn Predictor")

tab_predict, tab_insights = st.tabs(["🔮 Predict", "📊 Data Insights"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab_predict:
    st.write("Fill in customer details to estimate the likelihood they'll churn.")

    with st.form("churn_form"):
        st.subheader("Customer Profile")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col2:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)
            cltv = st.number_input("CLTV (Customer Lifetime Value score)", min_value=0, max_value=10000, value=4000)

        st.subheader("Services")
        col3, col4 = st.columns(2)
        with col3:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        with col4:
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        st.subheader("Account")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Build a raw-format single-row dataframe — SAME shape as training data
        # before any encoding. The pipeline handles all preprocessing internally.
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Senior Citizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure Months': tenure,
            'Phone Service': phone_service,
            'Multiple Lines': multiple_lines,
            'Internet Service': internet_service,
            'Online Security': online_security,
            'Online Backup': online_backup,
            'Device Protection': device_protection,
            'Tech Support': tech_support,
            'Streaming TV': streaming_tv,
            'Streaming Movies': streaming_movies,
            'Contract': contract,
            'Paperless Billing': paperless,
            'Payment Method': payment_method,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges,
            'CLTV': cltv,
        }])

        # binary columns need the same Yes/No -> 0/1 mapping used at training time
        binary_maps = {
            'Senior Citizen': {'No': 0, 'Yes': 1},
            'Gender': {'Male': 1, 'Female': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'Phone Service': {'Yes': 1, 'No': 0},
            'Paperless Billing': {'Yes': 1, 'No': 0},
        }
        for col, mapping in binary_maps.items():
            input_df[col] = input_df[col].map(mapping)

        proba = pipeline.predict_proba(input_df)[0][1]
        prediction = pipeline.predict(input_df)[0]

        st.divider()
        st.subheader("Result")
        if prediction == 1:
            st.error(f"⚠️ High churn risk — estimated probability: **{proba:.1%}**")
        else:
            st.success(f"✅ Likely to stay — estimated churn probability: **{proba:.1%}**")
        st.progress(min(proba, 1.0))
        st.caption("This is a model estimate based on the trained Random Forest classifier, not a guarantee.")

        # --- SHAP explanation ---
        st.divider()
        st.subheader("Why this prediction?")

        transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contrib = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)
        top_contrib = contrib.head(8)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#d62728' if v > 0 else '#2ca02c' for v in top_contrib.values]
        ax.barh(top_contrib.index[::-1], top_contrib.values[::-1], color=colors[::-1])
        ax.set_xlabel("Impact on churn probability")
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Red bars push toward churn, green bars push toward staying.")

# ============================================================
# TAB 2: DATA INSIGHTS (EDA, computed live from the dataset)
# ============================================================
with tab_insights:
    st.write("Key patterns found in the training data that explain what drives churn.")

    @st.cache_data
    def load_eda_data():
        df = pd.read_excel("Telco_customer_churn.xlsx")
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df.dropna(subset=['Total Charges'], inplace=True)
        return df

    df = load_eda_data()

    # --- Headline metrics ---
    total_customers = len(df)
    churn_rate = (df['Churn Value'].mean()) if 'Churn Value' in df.columns else (df['Churn Label'] == 'Yes').mean()
    avg_tenure = df['Tenure Months'].mean()
    avg_monthly = df['Monthly Charges'].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{total_customers:,}")
    m2.metric("Churn Rate", f"{churn_rate:.1%}")
    m3.metric("Avg. Tenure", f"{avg_tenure:.0f} months")
    m4.metric("Avg. Monthly Charge", f"${avg_monthly:.0f}")

    st.divider()

    col_a, col_b = st.columns(2)

    # --- Churn rate by tenure group (the U-shape insight) ---
    with col_a:
        st.subheader("Churn Rate by Tenure")
        df_eda = df.copy()
        df_eda['Tenure Group'] = pd.cut(
            df_eda['Tenure Months'], bins=[0, 12, 24, 36, 48, 60, 72],
            labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
        )
        churn_by_tenure = df_eda.groupby('Tenure Group', observed=True)['Churn Value'].mean()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(churn_by_tenure.index.astype(str), churn_by_tenure.values, color='#e07a5f')
        ax1.set_ylabel("Churn Rate")
        ax1.set_xlabel("Tenure (months)")
        ax1.set_title("New and long-tenure customers behave very differently")
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("Churn is highest for brand-new customers, drops through the middle, and rarely spikes again for long-tenure customers.")

    # --- Churn rate by contract type ---
    with col_b:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby('Contract', observed=True)['Churn Value'].mean().sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(contract_churn.index, contract_churn.values, color='#3d5a80')
        ax2.set_ylabel("Churn Rate")
        ax2.set_title("Contract length is the single strongest churn lever")
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("Month-to-month customers churn far more than annual or biennial contract holders.")

    st.divider()

    col_c, col_d = st.columns(2)

    # --- Monthly charges distribution by churn status ---
    with col_c:
        st.subheader("Monthly Charges by Churn Status")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Churn Label', y='Monthly Charges', data=df, ax=ax3, palette=['#81b29a', '#e07a5f'])
        ax3.set_title("Churned customers tend to pay more per month")
        plt.tight_layout()
        st.pyplot(fig3)

    # --- Top reasons customers actually left ---
    with col_d:
        st.subheader("Top Stated Reasons for Leaving")
        if 'Churn Reason' in df.columns:
            top_reasons = df['Churn Reason'].dropna().value_counts().head(8)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.barh(top_reasons.index[::-1], top_reasons.values[::-1], color='#9c6644')
            ax4.set_xlabel("Number of customers")
            ax4.set_title("Why churned customers said they left")
            plt.tight_layout()
            st.pyplot(fig4)
            st.caption("Only available for customers who actually churned — excluded from the model itself to avoid leakage.")
        else:
            st.info("Churn Reason column not found in this dataset.")

    st.divider()

    # --- Feature importance from the trained model ---
    st.subheader("What the Model Learned Matters Most")
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.barh(importances.index[::-1], importances.values[::-1], color='#4a5759')
    ax5.set_xlabel("Feature Importance")
    ax5.set_title("Top factors the Random Forest relies on for predictions")
    plt.tight_layout()
    st.pyplot(fig5)
    st.caption("This lines up with the EDA — contract type and tenure dominate, confirming the patterns above weren't coincidental.")