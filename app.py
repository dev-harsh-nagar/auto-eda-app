import io
import os
import pandas as pd
import streamlit as st
from litellm import completion

st.set_page_config(page_title="Auto-EDA Tool", layout="wide")
st.title("📊 Auto-EDA & AI Profiler")

# 1. CSV File Upload
uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📌 Dataset Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", f"{df.isna().sum().sum()}")
    col4.metric("Duplicates", f"{df.duplicated().sum()}")

    # 2. Automated Quality Flags
    st.subheader("🚩 Data Quality Flags")
    alerts = []
    
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            alerts.append(f"Column `{col}` has {count} missing values ({count/len(df):.1%}).")

    for col in df.columns:
        if df[col].nunique() == 1:
            alerts.append(f"Column `{col}` has only 1 unique value (zero variance).")
        elif df[col].nunique() == len(df) and df[col].dtype == 'object':
            alerts.append(f"Column `{col}` has all unique string values (high cardinality/ID column).")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No major quality warnings detected!")

    # 3. Data Preview
    with st.expander("🔍 View Raw Data Sample & Summary Statistics"):
        st.dataframe(df.head(10))
        st.write(df.describe(include="all").transpose())

# 4. AI Summary using Google Gemini
    st.subheader("🤖 AI Executive Summary")
    import google.generativeai as genai

    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        api_key = st.text_input("Enter Gemini API Key (Fallback)", type="password")

    if st.button("Generate AI Insights"):
        if not api_key:
            st.error("Please provide a Gemini API Key.")
        else:
            try:
                genai.configure(api_key=api_key)
                
                # Fetch available models directly from your key's project permissions
                available_models = [
                    m.name for m in genai.list_models() 
                    if 'generateContent' in m.supported_generation_methods
                ]
                
                # Auto-select gemini-2.5-flash if available, otherwise pick the first valid model
                # Gemini 1.5,2.0 & 2.5 wasn't working properly, so I have added gemini-3.6-flash
                target_model = next((m for m in available_models if "gemini-3.6-flash" in m), available_models[0])
                
                model = genai.GenerativeModel(target_model)
                
                prompt = f"""
                Summarize this dataset for an executive:
                - Shape: {df.shape[0]} rows, {df.shape[1]} columns
                - Quality Flags: {', '.join(alerts) if alerts else 'None'}
                - Columns & Types: {dict(df.dtypes.astype(str))}
                - Sample Data:
                {df.head(3).to_string()}
                
                Provide 3 sections: 1. Core Overview, 2. Major Risks/Quality Issues, 3. Suggested Next Steps.
                """
                
                with st.spinner(f"Analyzing dataset with {target_model}..."):
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"Error calling Gemini API: {e}")

    
