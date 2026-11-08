"""
Interactive Type 2 Diabetes Risk Prediction Tool
Uses trained Logistic Regression / Random Forest models based on BRFSS 2014 features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from project_paths import PROCESSED_CSV_PATH

def get_trained_pipeline():
    print("Preparing and training quick baseline model (Logistic Regression)...")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X = df.drop(columns=['DIABETES_T2'])
    y = df['DIABETES_T2']

    categorical_cols = [
        'GEN_HEALTH', 'AGE_GRP', 'BMI_CAT', 'CHECKUP_FREQ', 'INCOME_GRP', 'RACE_GRP',
        'METRO_STATUS', 'EMPLOYMENT', 'MARITAL_STATUS', 'EDUCATION', 'SLEEP_GRP',
        'HEALTH_COVERAGE_SRC', 'HOME_OWNERSHIP', 'SMOKING_STATUS'
    ]
    binary_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('bin', 'passthrough', binary_cols)
        ]
    )

    X_enc = preprocessor.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_smote, y_smote)
    return preprocessor, scaler, clf

def predict_sample(sample_dict, preprocessor, scaler, model):
    df_sample = pd.DataFrame([sample_dict])
    enc = preprocessor.transform(df_sample)
    scaled = scaler.transform(enc)
    prob = model.predict_proba(scaled)[0, 1]
    prediction = int(prob >= 0.5)
    return prediction, prob

if __name__ == "__main__":
    if not PROCESSED_CSV_PATH.exists():
        print("Please run preprocess.py first.")
        exit(1)

    preprocessor, scaler, model = get_trained_pipeline()

    # Sample Profile 1: High Risk Profile
    # Older, obese, poor general health, checks doctor frequently, sleeps >9 hours, coronary heart disease
    sample_high_risk = {
        'GEN_HEALTH': 5, # Poor
        'AGE_GRP': 5,    # 71-80 y
        'BMI_CAT': 4,    # Obese
        'CHECKUP_FREQ': 1, # <1 year (new risk factor)
        'INCOME_GRP': 2, # $10-15k
        'RACE_GRP': 1,   # White
        'METRO_STATUS': 1,
        'FLU_SHOT': 1,
        'EMPLOYMENT': 7, # Retired
        'SEX': 1,        # Male
        'MARITAL_STATUS': 1, # Married
        'EDUCATION': 2,
        'SLEEP_GRP': 3,  # >=9 hours (new risk factor)
        'HEART_DISEASE': 1, # Yes
        'HEALTH_COVERAGE_SRC': 3, # Medicare
        'MENTAL_HEALTH_GRP': 1,
        'KIDNEY_DISEASE': 1,
        'USE_EQUIP': 1,
        'PHYSICAL_ACTIVITY': 0, # No
        'DEPRESSION': 0,
        'HOME_OWNERSHIP': 1,
        'EXERCISE': 0,
        'BLIND_VISION_DIFFICULTY': 0,
        'COGNITIVE_DIFFICULTY': 0,
        'ANY_HEALTHCARE': 1,
        'SMOKING_STATUS': 3
    }

    # Sample Profile 2: Low Risk Profile
    # Younger adult, normal weight, excellent health, active, 7-8h sleep, no chronic disease
    sample_low_risk = {
        'GEN_HEALTH': 1, # Excellent
        'AGE_GRP': 1,    # 31-40 y
        'BMI_CAT': 2,    # Normal weight
        'CHECKUP_FREQ': 3, # 3-5 years
        'INCOME_GRP': 8, # >$75k
        'RACE_GRP': 1,
        'METRO_STATUS': 3,
        'FLU_SHOT': 0,
        'EMPLOYMENT': 1, # Employed
        'SEX': 0,        # Female
        'MARITAL_STATUS': 1,
        'EDUCATION': 4, # College graduate
        'SLEEP_GRP': 2,  # 7-8 hours
        'HEART_DISEASE': 0,
        'HEALTH_COVERAGE_SRC': 1, # Employer
        'MENTAL_HEALTH_GRP': 1,
        'KIDNEY_DISEASE': 0,
        'USE_EQUIP': 0,
        'PHYSICAL_ACTIVITY': 1,
        'DEPRESSION': 0,
        'HOME_OWNERSHIP': 1,
        'EXERCISE': 1,
        'BLIND_VISION_DIFFICULTY': 0,
        'COGNITIVE_DIFFICULTY': 0,
        'ANY_HEALTHCARE': 1,
        'SMOKING_STATUS': 4 # Never smoked
    }

    print("\n--- Testing High Risk Patient Sample ---")
    pred1, prob1 = predict_sample(sample_high_risk, preprocessor, scaler, model)
    print(f"Risk Score: {prob1*100:.2f}% | Prediction: {'HIGH RISK (Type 2 Diabetes)' if pred1 else 'LOW RISK'}")

    print("\n--- Testing Low Risk Patient Sample ---")
    pred2, prob2 = predict_sample(sample_low_risk, preprocessor, scaler, model)
    print(f"Risk Score: {prob2*100:.2f}% | Prediction: {'HIGH RISK (Type 2 Diabetes)' if pred2 else 'LOW RISK'}")
