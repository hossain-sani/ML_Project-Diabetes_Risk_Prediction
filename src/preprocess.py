"""
Data Preprocessing Script for:
Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques
(Xie et al., Preventing Chronic Disease, CDC, 2019)
Using 2014 CDC BRFSS Data
"""

import os
import pyreadstat
import pandas as pd
import numpy as np
from project_paths import PROCESSED_CSV_PATH, RAW_XPT_PATH

def preprocess_brfss2014(xpt_path=RAW_XPT_PATH, output_csv=PROCESSED_CSV_PATH):
    if not os.path.exists(xpt_path):
        raise FileNotFoundError(f"Input file {xpt_path} not found. Please run download_data.py first.")

    target_vars = [
        'GENHLTH', '_AGEG5YR', '_BMI5CAT', 'CHECKUP1', 'INCOME2', '_RACE', 'MSCODE', 
        'FLUSHOT6', 'EMPLOY1', 'SEX', 'MARITAL', '_EDUCAG', 'SLEPTIM1', 'CVDCRHD4', 
        'HLTHCVR1', 'MENTHLTH', 'CHCKIDNY', 'USEEQUIP', '_TOTINDA', 'ADDEPEV2', 
        'RENTHOM1', 'EXERANY2', 'BLIND', 'DECIDE', 'HLTHPLN1', 'DIABETE3', '_SMOKER3',
        'DIABAGE2'
    ]

    print(f"Reading target variables from {xpt_path}...")
    df, _ = pyreadstat.read_xport(xpt_path, usecols=target_vars)
    print(f"Loaded raw records: {len(df)}")

    # 1. Target Variable: DIABETE3
    # 1 = Yes (Diabetes), 3 = No.
    # Exclude gestational diabetes (2), prediabetes (4), refused/missing (7, 9, NaN)
    df = df[df['DIABETE3'].isin([1.0, 3.0])].copy()

    # 2. Age criteria & Exclude Type 1 Diabetes:
    # Respondents younger than 30 excluded (categories 1 and 2 in _AGEG5YR)
    # Among positive cases, age of diagnosis must be >= 30 (DIABAGE2 >= 30) to exclude Type 1
    df = df[df['_AGEG5YR'].isin(range(3, 14))].copy()
    
    # Exclude positive respondents diagnosed under 30 (DIABAGE2 < 30)
    type1_mask = (df['DIABETE3'] == 1.0) & (df['DIABAGE2'] < 30)
    df = df[~type1_mask].copy()

    # Create binary target: 1 = Type 2 Diabetes, 0 = No Diabetes
    df['DIABETES_T2'] = (df['DIABETE3'] == 1.0).astype(int)

    # Recode Age into 6 condensed groups (as per Table 2 & Appendix):
    # 1: 31-40 y, 2: 41-50 y, 3: 51-60 y, 4: 61-70 y, 5: 71-80 y, 6: >80 y
    age_map = {3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4, 11: 5, 12: 5, 13: 6}
    df['AGE_GRP'] = df['_AGEG5YR'].map(age_map)

    # 3. BMI categories: 1: Underweight, 2: Normal, 3: Overweight, 4: Obese
    df = df[df['_BMI5CAT'].isin([1, 2, 3, 4])].copy()
    df['BMI_CAT'] = df['_BMI5CAT'].astype(int)

    # 4. Checkup frequency: 1: <1 y, 2: 1-2 y, 3: 3-5 y, 4: >5 y, 8: Never (recode to 5: Never)
    df = df[df['CHECKUP1'].isin([1, 2, 3, 4, 8])].copy()
    df['CHECKUP_FREQ'] = df['CHECKUP1'].replace({8: 5}).astype(int)

    # 5. Income: 1 to 8 (exclude 77: Don't know, 99: Refused, NaN)
    df = df[df['INCOME2'].isin(range(1, 9))].copy()
    df['INCOME_GRP'] = df['INCOME2'].astype(int)

    # 6. Race: 1 to 8 (exclude 9: Refused, NaN)
    df = df[df['_RACE'].isin(range(1, 9))].copy()
    df['RACE_GRP'] = df['_RACE'].astype(int)

    # 7. Metropolitan status: 1: City center, 2: County, 3: Suburban, 5: Not in MSA (recode 5 -> 4)
    df = df[df['MSCODE'].isin([1, 2, 3, 5])].copy()
    df['METRO_STATUS'] = df['MSCODE'].replace({5: 4}).astype(int)

    # 8. Flu shot: 1: Yes, 2: No
    df = df[df['FLUSHOT6'].isin([1, 2])].copy()
    df['FLU_SHOT'] = (df['FLUSHOT6'] == 1).astype(int)

    # 9. Employment: 1 to 8
    df = df[df['EMPLOY1'].isin(range(1, 9))].copy()
    df['EMPLOYMENT'] = df['EMPLOY1'].astype(int)

    # 10. Sex: 1: Male, 2: Female
    df = df[df['SEX'].isin([1, 2])].copy()
    df['SEX'] = (df['SEX'] == 1).astype(int) # 1 = Male, 0 = Female

    # 11. Marital status: 1 to 6
    df = df[df['MARITAL'].isin(range(1, 7))].copy()
    df['MARITAL_STATUS'] = df['MARITAL'].astype(int)

    # 12. Education level: 1 to 4
    df = df[df['_EDUCAG'].isin(range(1, 5))].copy()
    df['EDUCATION'] = df['_EDUCAG'].astype(int)

    # 13. Sleep hours: condensed into 1: <=6h, 2: 7-8h, 3: >=9h
    df = df[df['SLEPTIM1'].between(1, 24)].copy()
    def recode_sleep(val):
        if val <= 6:
            return 1
        elif val in [7, 8]:
            return 2
        else:
            return 3
    df['SLEEP_GRP'] = df['SLEPTIM1'].apply(recode_sleep).astype(int)

    # 14. Angina or Coronary Heart Disease: 1: Yes, 2: No
    df = df[df['CVDCRHD4'].isin([1, 2])].copy()
    df['HEART_DISEASE'] = (df['CVDCRHD4'] == 1).astype(int)

    # 15. Primary health care coverage: 1 to 8
    df = df[df['HLTHCVR1'].isin(range(1, 9))].copy()
    df['HEALTH_COVERAGE_SRC'] = df['HLTHCVR1'].astype(int)

    # 16. Mental health: 1: 0-5 days, 2: 6-30 days
    df = df[df['MENTHLTH'].isin([88] + list(range(1, 31)))].copy()
    df['MENTAL_HEALTH_GRP'] = df['MENTHLTH'].apply(lambda v: 1 if v == 88 or v <= 5 else 2).astype(int)

    # 17. Kidney disease: 1: Yes, 2: No
    df = df[df['CHCKIDNY'].isin([1, 2])].copy()
    df['KIDNEY_DISEASE'] = (df['CHCKIDNY'] == 1).astype(int)

    # 18. Use special equipment (cane, wheelchair): 1: Yes, 2: No
    df = df[df['USEEQUIP'].isin([1, 2])].copy()
    df['USE_EQUIP'] = (df['USEEQUIP'] == 1).astype(int)

    # 19. Physical activity (calculated): 1: Yes, 2: No
    df = df[df['_TOTINDA'].isin([1, 2])].copy()
    df['PHYSICAL_ACTIVITY'] = (df['_TOTINDA'] == 1).astype(int)

    # 20. Depression disorder: 1: Yes, 2: No
    df = df[df['ADDEPEV2'].isin([1, 2])].copy()
    df['DEPRESSION'] = (df['ADDEPEV2'] == 1).astype(int)

    # 21. Home ownership: 1: Own, 2: Rent, 3: Other
    df = df[df['RENTHOM1'].isin([1, 2, 3])].copy()
    df['HOME_OWNERSHIP'] = df['RENTHOM1'].astype(int)

    # 22. Exercise outside of work: 1: Yes, 2: No
    df = df[df['EXERANY2'].isin([1, 2])].copy()
    df['EXERCISE'] = (df['EXERANY2'] == 1).astype(int)

    # 23. Blind or serious difficulty seeing: 1: Yes, 2: No
    df = df[df['BLIND'].isin([1, 2])].copy()
    df['BLIND_VISION_DIFFICULTY'] = (df['BLIND'] == 1).astype(int)

    # 24. Difficulty concentrating / deciding: 1: Yes, 2: No
    df = df[df['DECIDE'].isin([1, 2])].copy()
    df['COGNITIVE_DIFFICULTY'] = (df['DECIDE'] == 1).astype(int)

    # 25. Any health care coverage: 1: Yes, 2: No
    df = df[df['HLTHPLN1'].isin([1, 2])].copy()
    df['ANY_HEALTHCARE'] = (df['HLTHPLN1'] == 1).astype(int)

    # 26. General health: 1: Excellent, 2: Very good, 3: Good, 4: Fair, 5: Poor
    df = df[df['GENHLTH'].isin(range(1, 6))].copy()
    df['GEN_HEALTH'] = df['GENHLTH'].astype(int)

    # 27. Smoking status: 1: Every day, 2: Some days, 3: Former smoker, 4: Never smoked
    df = df[df['_SMOKER3'].isin(range(1, 5))].copy()
    df['SMOKING_STATUS'] = df['_SMOKER3'].astype(int)

    # Select final standardized feature set
    feature_columns = [
        'GEN_HEALTH', 'AGE_GRP', 'BMI_CAT', 'CHECKUP_FREQ', 'INCOME_GRP', 'RACE_GRP',
        'METRO_STATUS', 'FLU_SHOT', 'EMPLOYMENT', 'SEX', 'MARITAL_STATUS', 'EDUCATION',
        'SLEEP_GRP', 'HEART_DISEASE', 'HEALTH_COVERAGE_SRC', 'MENTAL_HEALTH_GRP',
        'KIDNEY_DISEASE', 'USE_EQUIP', 'PHYSICAL_ACTIVITY', 'DEPRESSION', 'HOME_OWNERSHIP',
        'EXERCISE', 'BLIND_VISION_DIFFICULTY', 'COGNITIVE_DIFFICULTY', 'ANY_HEALTHCARE',
        'SMOKING_STATUS'
    ]
    target_column = 'DIABETES_T2'

    final_df = df[feature_columns + [target_column]].copy()

    total_count = len(final_df)
    pos_count = final_df[target_column].sum()
    neg_count = total_count - pos_count

    print(f"\n================ PREPROCESSING COMPLETE ================")
    print(f"Total respondents retained: {total_count:,} (Expected: 138,146)")
    print(f"Type 2 Diabetes positive:   {pos_count:,} ({pos_count/total_count*100:.2f}%) (Expected: 20,467 / 14.8%)")
    print(f"Non-diabetic negative:     {neg_count:,} ({neg_count/total_count*100:.2f}%) (Expected: 117,679 / 85.2%)")
    print(f"Features: {len(feature_columns)}")
    print(f"========================================================\n")

    print(f"Saving preprocessed data to {output_csv}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} ({os.path.getsize(output_csv) / (1024*1024):.2f} MB).")
    return final_df

if __name__ == "__main__":
    preprocess_brfss2014()
