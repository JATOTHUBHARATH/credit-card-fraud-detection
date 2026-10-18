import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Pick a sample transaction and see whether the XGBoost model flags it.")

model = joblib.load("xgb_model.pkl")
demo = pd.read_csv("demo_transactions.csv")

threshold = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05)
idx = st.number_input("Transaction number", 0, len(demo) - 1, 0)

row = demo.iloc[[idx]]
actual = int(row["Class"].iloc[0])
prob = model.predict_proba(row.drop("Class", axis=1))[0, 1]

st.metric("Fraud probability", f"{prob:.1%}")
if prob >= threshold:
    st.error("🚨 Flagged as FRAUD")
else:
    st.success("✅ Looks NORMAL")
st.caption(f"Actual label: {'Fraud' if actual == 1 else 'Normal'}")

with st.expander("Show transaction features"):
    st.dataframe(row.drop("Class", axis=1).T)