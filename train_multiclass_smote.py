print("Multi-Class IDS with SMOTE Started!")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------
# LOAD DATA
# -------------------------
print("Loading CSV files...")
df1 = pd.read_csv("UNSW_NB15_training-set.csv")
df2 = pd.read_csv("UNSW_NB15_testing-set.csv")

df = pd.concat([df1, df2], ignore_index=True)
print("Combined data:", df.shape)

# -------------------------
# TARGET = attack category (10 types)
# -------------------------
print("Preparing target labels (multi-class)...")
y = df['attack_cat']
X = df.drop(['label','attack_cat'], axis=1)

# Encode attack category names
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
classes = label_enc.classes_
print("Attack types:", list(classes))

# -------------------------
# ONE-HOT ENCODE INPUT
# -------------------------
print("One-hot encoding features...")
X = pd.get_dummies(X)

# -------------------------
# TRAIN/TEST SPLIT
# -------------------------
print("Splitting 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Shapes:", X_train.shape, X_test.shape)

# -------------------------
# APPLY SMOTE
# -------------------------
print("Applying SMOTE oversampling...")
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
print("Balanced train shape:", X_train.shape)

# -------------------------
# SCALE
# -------------------------
print("Scaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------
# TRAIN MODEL
# -------------------------
print(" Training XGBoost Multi-class...")
model = XGBClassifier(
    n_estimators=450,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)
model.fit(X_train, y_train)
print("Training complete!")

# -------------------------
# EVALUATE
# -------------------------
print("Evaluating model...\n")
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# -------------------------
# SAVE MODEL + MAPPINGS
# -------------------------
print("\nSaving model, scaler, and columns...")
joblib.dump(model, "ids_multiclass_model.pkl")
joblib.dump(scaler, "ids_multiclass_scaler.pkl")
joblib.dump(X.columns, "ids_multiclass_columns.pkl")
joblib.dump(classes, "ids_attack_classes.pkl")

print("DONE! Files created:")
print(" - ids_multiclass_model.pkl")
print(" - ids_multiclass_scaler.pkl")
print(" - ids_multiclass_columns.pkl")
print(" - ids_attack_classes.pkl")
