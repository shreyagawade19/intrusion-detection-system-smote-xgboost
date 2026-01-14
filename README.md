# Intrusion Detection System (UNSW-NB15 + XGBoost + SMOTE) #

A machine-learning Intrusion Detection System trained on the UNSW-NB15 cybersecurity dataset to classify network traffic into 10 attack types.

## Features ##
1. Multi-class classification (Normal + 9 attacks).
2. XGBoost model trained with SMOTE to fix class imbalance.
3. Preprocessing: One-hot encoding + feature scaling.
4. Saved model ready for reuse (no retraining needed).
5. Predict custom network packets using Python scripts.

## Main Files ##
train_multiclass_smote.py   → Train model
predict_multiclass.py       → Predict attack category from sample input
predict_again.py            → Load saved model and predict
ids_multiclass_model.pkl    → Trained XGBoost IDS model
ids_multiclass_scaler.pkl   → StandardScaler object
ids_multiclass_columns.pkl  → Expected features list
ids_attack_classes.pkl      → Maps model output to attack name

Dataset files included:
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv

## Attack Classes Detected ##
Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Shellcode, Backdoor, Analysis, Worms

## How to run ##
1. Activate environment
venv\Scripts\activate
2. Predict using saved model
python predict_multiclass.py

## To retrain ##
python train_multiclass_smote.py

## Requirements ##
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib

## Credits ##
Dataset: UNSW-NB15, University of New South Wales (Canberra, Australia)

