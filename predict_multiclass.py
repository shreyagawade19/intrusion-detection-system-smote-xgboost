print("Loading Multi-Class IDS Model...")

import pandas as pd
import joblib

# Load components
model = joblib.load("ids_multiclass_model.pkl")
scaler = joblib.load("ids_multiclass_scaler.pkl")
columns = joblib.load("ids_multiclass_columns.pkl")
attack_labels = joblib.load("ids_attack_classes.pkl")

print("Model, scaler and metadata loaded!")

# --------------------------
# MODIFY YOUR PACKET HERE
# --------------------------
sample_packet = {
 'id': 47912,
 'dur': 0.000009,
 'proto': 'ddp',
 'service': '-',
 'state': 'INT',
 'spkts': 2,
 'dpkts': 0,
 'sbytes': 200,
 'dbytes': 0,
 'rate': 111111.1072,
 'sttl': 254,
 'dttl': 0,
 'sload': 88888888.0,
 'dload': 0.0,
 'sloss': 0,
 'dloss': 0,
 'sinpkt': 0.009,
 'dinpkt': 0.0,
 'sjit': 0.0,
 'djit': 0.0,
 'swin': 0,
 'stcpb': 0,
 'dtcpb': 0,
 'dwin': 0,
 'tcprtt': 0.0,
 'synack': 0.0,
 'ackdat': 0.0,
 'smean': 100,
 'dmean': 0,
 'trans_depth': 0,
 'response_body_len': 0,
 'ct_srv_src': 4,
 'ct_state_ttl': 2,
 'ct_dst_ltm': 1,
 'ct_src_dport_ltm': 1,
 'ct_dst_sport_ltm': 1,
 'ct_dst_src_ltm': 4,
 'is_ftp_login': 0,
 'ct_ftp_cmd': 0,
 'ct_flw_http_mthd': 0,
 'ct_src_ltm': 2,
 'ct_srv_dst': 4,
 'is_sm_ips_ports': 0
}


print("\nInput Packet:")
print(sample_packet)

# Convert to dataframe
df = pd.DataFrame([sample_packet])

# One-hot encode + align to training columns
df = pd.get_dummies(df)
df = df.reindex(columns=columns, fill_value=0)

# Scale numeric columns
df = scaler.transform(df)

# Predict class index
pred = model.predict(df)[0]

attack_name = attack_labels[pred]

print("\nPrediction:")
print(f"➡ Attack Category: {attack_name}")
