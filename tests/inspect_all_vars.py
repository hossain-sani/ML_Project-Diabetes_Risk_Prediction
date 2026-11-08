import pyreadstat
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

target_vars = [
    'GENHLTH', '_AGEG5YR', '_BMI5CAT', 'CHECKUP1', 'INCOME2', '_RACE', 'MSCODE', 
    'FLUSHOT6', 'EMPLOY1', 'SEX', 'MARITAL', '_EDUCAG', 'SLEPTIM1', 'CVDCRHD4', 
    'HLTHCVR1', 'MENTHLTH', 'CHCKIDNY', 'USEEQUIP', '_TOTINDA', 'ADDEPEV2', 
    'RENTHOM1', 'EXERANY2', 'BLIND', 'DECIDE', 'HLTHPLN1', 'DIABETE3', '_SMOKER3'
]

df, meta = pyreadstat.read_xport(root_dir / "data" / "raw" / "LLCP2014.XPT", usecols=target_vars)

for var in target_vars:
    vc = df[var].value_counts(dropna=False).sort_index()
    print(f"\n--- {var} ---")
    print(dict(vc))
