import pyreadstat
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

target_vars = [
    'GENHLTH', '_AGEG5YR', '_BMI5CAT', 'CHECKUP1', 'INCOME2', '_RACE', 'MSCODE', 
    'FLUSHOT6', 'EMPLOY1', 'SEX', 'MARITAL', '_EDUCAG', 'SLEPTIM1', 'CVDCRHD4', 
    'HLTHCVR1', 'MENTHLTH', 'CHCKIDNY', 'USEEQUIP', '_TOTINDA', 'ADDEPEV2', 
    'RENTHOM1', 'EXERANY2', 'BLIND', 'DECIDE', 'HLTHPLN1', 'DIABETE3', '_SMOKER3'
]

print("Reading data...")
df, _ = pyreadstat.read_xport(root_dir / "data" / "raw" / "LLCP2014.XPT", usecols=target_vars)

# 1. DIABETE3: 1 is Type 2 Diabetes, 3 is No Diabetes. Exclude 2, 4, 7, 9, NaN
df = df[df['DIABETE3'].isin([1.0, 3.0])].copy()
df['DIABETE3'] = (df['DIABETE3'] == 1.0).astype(int)

# 2. _AGEG5YR: Exclude 1 (18-24), 2 (25-29), 14 (refused/missing)
df = df[df['_AGEG5YR'].isin(range(3, 14))].copy()

# Recode age into 6 groups:
# 3,4 -> 1 (31-40)
# 5,6 -> 2 (41-50)
# 7,8 -> 3 (51-60)
# 9,10 -> 4 (61-70)
# 11,12 -> 5 (71-80)
# 13 -> 6 (>80)
age_map = {3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4, 11: 5, 12: 5, 13: 6}
df['_AGEG5YR'] = df['_AGEG5YR'].map(age_map)

# 3. _BMI5CAT: 1, 2, 3, 4
df = df[df['_BMI5CAT'].isin([1, 2, 3, 4])].copy()

# 4. CHECKUP1: 1 (<1y), 2 (1-2y), 3 (3-5y), 4 (>5y), 8 (Never -> recode to 5 or 6 or 8)
# In BRFSS: 1,2,3,4, 8. 7 is Don't know, 9 is Refused
df = df[df['CHECKUP1'].isin([1, 2, 3, 4, 8])].copy()

# 5. INCOME2: 1 to 8
df = df[df['INCOME2'].isin(range(1, 9))].copy()

# 6. _RACE: 1 to 8
df = df[df['_RACE'].isin(range(1, 9))].copy()

# 7. MSCODE: 1, 2, 3, 5
df = df[df['MSCODE'].isin([1, 2, 3, 5])].copy()

# 8. FLUSHOT6: 1, 2
df = df[df['FLUSHOT6'].isin([1, 2])].copy()

# 9. EMPLOY1: 1 to 8
df = df[df['EMPLOY1'].isin(range(1, 9))].copy()

# 10. SEX: 1, 2
df = df[df['SEX'].isin([1, 2])].copy()

# 11. MARITAL: 1 to 6
df = df[df['MARITAL'].isin(range(1, 7))].copy()

# 12. _EDUCAG: 1 to 4
df = df[df['_EDUCAG'].isin(range(1, 5))].copy()

# 13. SLEPTIM1: 1 to 24
df = df[df['SLEPTIM1'].between(1, 24)].copy()
def recode_sleep(val):
    if val <= 6:
        return 1
    elif val in [7, 8]:
        return 2
    else:
        return 3
df['SLEPTIM1'] = df['SLEPTIM1'].apply(recode_sleep)

# 14. CVDCRHD4: 1, 2
df = df[df['CVDCRHD4'].isin([1, 2])].copy()

# 15. HLTHCVR1: 1 to 8
df = df[df['HLTHCVR1'].isin(range(1, 9))].copy()

# 16. MENTHLTH: 88 (None) or 1 to 30
df = df[df['MENTHLTH'].isin([88] + list(range(1, 31)))].copy()
def recode_ment(val):
    if val == 88 or (1 <= val <= 5):
        return 1
    else:
        return 2
df['MENTHLTH'] = df['MENTHLTH'].apply(recode_ment)

# 17. CHCKIDNY: 1, 2
df = df[df['CHCKIDNY'].isin([1, 2])].copy()

# 18. USEEQUIP: 1, 2
df = df[df['USEEQUIP'].isin([1, 2])].copy()

# 19. _TOTINDA: 1, 2
df = df[df['_TOTINDA'].isin([1, 2])].copy()

# 20. ADDEPEV2: 1, 2
df = df[df['ADDEPEV2'].isin([1, 2])].copy()

# 21. RENTHOM1: 1, 2, 3
df = df[df['RENTHOM1'].isin([1, 2, 3])].copy()

# 22. EXERANY2: 1, 2
df = df[df['EXERANY2'].isin([1, 2])].copy()

# 23. BLIND: 1, 2
df = df[df['BLIND'].isin([1, 2])].copy()

# 24. DECIDE: 1, 2
df = df[df['DECIDE'].isin([1, 2])].copy()

# 25. HLTHPLN1: 1, 2
df = df[df['HLTHPLN1'].isin([1, 2])].copy()

# 26. GENHLTH: 1 to 5
df = df[df['GENHLTH'].isin(range(1, 6))].copy()

# 27. _SMOKER3: 1 to 4
df = df[df['_SMOKER3'].isin(range(1, 5))].copy()

print(f"\nFinal dataset shape: {df.shape}")
print(f"Total respondents: {len(df)}")
pos = df['DIABETE3'].sum()
print(f"Diabetes positive: {pos} ({pos/len(df)*100:.2f}%)")
print(f"Diabetes negative: {len(df) - pos} ({(len(df)-pos)/len(df)*100:.2f}%)")
print(f"Expected in paper: 138,146 total, 20,467 positive (14.81%)")
