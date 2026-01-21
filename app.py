import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
model = joblib.load("ids_multiclass_model.pkl")
scaler = joblib.load("ids_multiclass_scaler.pkl")
columns = joblib.load("ids_multiclass_columns.pkl")
attack_map = joblib.load("ids_attack_classes.pkl")

st.set_page_config(page_title="Intrusion Detection System", layout="centered")

st.title("🛡️ Intrusion Detection System")
st.write("UNSW-NB15 | XGBoost | SMOTE")

st.subheader("Enter Network Flow Details")

# --- User Inputs ---
proto = st.selectbox("Protocol", ["tcp", "udp", "icmp", "ddp"])
service = st.text_input("Service", "http")
state = st.text_input("State", "INT")
dur = st.number_input("Duration", 0.0, 10.0, 0.001)
sbytes = st.number_input("Source Bytes", 0, 1000000, 100)
dbytes = st.number_input("Destination Bytes", 0, 1000000, 0)
sttl = st.number_input("Source TTL", 0, 255, 64)
dttl = st.number_input("Destination TTL", 0, 255, 1)

sample = {
    "proto": proto,
    "service": service,
    "state": state,
    "dur": dur,
    "sbytes": sbytes,
    "dbytes": dbytes,
    "sttl": sttl,
    "dttl": dttl
}

if st.button("🔍 Detect Attack"):
    df = pd.DataFrame([sample])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    attack = attack_map[pred]

    if attack == "Normal":
        st.success(f"🟢 Traffic is NORMAL")
    else:
        st.error(f"🚨 Attack Detected: **{attack}**")
