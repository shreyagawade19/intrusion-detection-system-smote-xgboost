import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD MODEL ARTIFACTS --------------------
model = joblib.load("ids_multiclass_model.pkl")
scaler = joblib.load("ids_multiclass_scaler.pkl")
columns = joblib.load("ids_multiclass_columns.pkl")
attack_map = joblib.load("ids_attack_classes.pkl")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="centered"
)

# -------------------- CONSTANTS --------------------
SEVERITY = {
    "Normal": ("🟢 NORMAL", "success"),
    "Reconnaissance": ("🟡 MEDIUM RISK", "warning"),
    "Analysis": ("🟡 MEDIUM RISK", "warning"),
    "DoS": ("🟠 HIGH RISK", "error"),
    "Fuzzers": ("🟠 HIGH RISK", "error"),
    "Exploits": ("🟠 HIGH RISK", "error"),
    "Backdoor": ("🔴 CRITICAL", "error"),
    "Shellcode": ("🔴 CRITICAL", "error"),
    "Worms": ("🔴 CRITICAL", "error"),
}

ATTACK_DESC = {
    "Normal": "No suspicious activity detected.",
    "DoS": "Denial of Service attack flooding system resources.",
    "Backdoor": "Hidden unauthorized access detected.",
    "Reconnaissance": "Scanning or information gathering behavior.",
    "Fuzzers": "Malformed or random input attack.",
    "Exploits": "Known vulnerability exploitation attempt.",
    "Shellcode": "Malicious payload execution attempt.",
    "Worms": "Self-propagating malware behavior.",
    "Analysis": "Traffic showing abnormal analysis patterns."
}

# Preset attack samples (from real UNSW-style patterns)
BACKDOOR_SAMPLE = {
    "proto": "ddp",
    "service": "-",
    "state": "INT",
    "dur": 0.000009,
    "sbytes": 200,
    "dbytes": 0,
    "sttl": 254,
    "dttl": 0
}

DOS_SAMPLE = {
    "proto": "tcp",
    "service": "http",
    "state": "S0",
    "dur": 0.000001,
    "sbytes": 0,
    "dbytes": 0,
    "sttl": 1,
    "dttl": 1
}

# -------------------- UI --------------------
st.title("🛡️ Intrusion Detection System")
st.caption("UNSW-NB15 | XGBoost | SMOTE | Streamlit Deployment")

tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Statistics", "ℹ About"])

# -------------------- TAB 1 : DETECTION --------------------
with tab1:
    st.subheader("Enter Network Flow Details")

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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Test Backdoor Sample"):
            sample = BACKDOOR_SAMPLE
    with col2:
        if st.button("🧪 Test DoS Sample"):
            sample = DOS_SAMPLE

    if st.button("🔍 Detect Attack"):
        df = pd.DataFrame([sample])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)[0]
        attack = attack_map[pred]

        probs = model.predict_proba(df_scaled)[0]
        confidence = max(probs) * 100

        label, level = SEVERITY.get(attack, ("⚠ UNKNOWN", "warning"))

        if level == "success":
            st.success(f"{label}")
        elif level == "warning":
            st.warning(f"{label}: {attack}")
        else:
            st.error(f"{label}: {attack}")

        st.metric("Detection Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)
        st.info(ATTACK_DESC.get(attack, "Unknown traffic behavior detected."))

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(attack)

# -------------------- TAB 2 : STATISTICS --------------------
with tab2:
    st.subheader("Attack Detection Statistics")

    if "history" in st.session_state and len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history, columns=["Attack Type"])
        chart = hist_df.value_counts().reset_index()
        chart.columns = ["Attack Type", "Count"]
        st.bar_chart(chart.set_index("Attack Type"))
    else:
        st.write("No detections yet.")

# -------------------- TAB 3 : ABOUT --------------------
with tab3:
    st.markdown("""
    **Intrusion Detection System (IDS)**  
    - Trained on UNSW-NB15 dataset  
    - Multi-class classification (10 attack families)  
    - Handles class imbalance using SMOTE  
    - Deployed using Streamlit  

    **Use Case:**  
    Real-time traffic classification for SOC monitoring, research, and demos.
    """)

st.markdown("---")
st.caption("Developed by Shreya Gawade | ML & Cybersecurity Project")
