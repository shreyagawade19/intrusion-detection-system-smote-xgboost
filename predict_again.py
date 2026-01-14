import pandas as pd
import joblib

# Load model artifacts
model = joblib.load("ids_multiclass_model.pkl")
scaler = joblib.load("ids_multiclass_scaler.pkl")
columns = joblib.load("ids_multiclass_columns.pkl")
attack_map = joblib.load("ids_attack_classes.pkl")

# Example packet (replace with your input)
sample_packet = {
 'srcip': '10.0.0.5',
 'sport': 3333,
 'dstip': '10.0.0.10',
 'dsport': 80,
 'proto': 'tcp',
 'state': 'SYN',
 'dur': 0.01,
 'sbytes': 100,
 'dbytes': 1000,
 'sttl': 64,
 'dttl': 1,
 'sloss': 0,
 'dloss': 0,
 'service': 'http'
}

# Convert to DataFrame
df = pd.DataFrame([sample_packet])

# One-hot encoding
df = pd.get_dummies(df)

# Align with training columns
df = df.reindex(columns=columns, fill_value=0)

# Scale numeric fields
df = scaler.transform(df)

# Predict
pred_index = model.predict(df)[0]
attack_name = attack_map[pred_index]

print("\n Prediction:", attack_name)
