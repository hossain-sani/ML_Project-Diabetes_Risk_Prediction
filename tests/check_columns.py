import pyreadstat
from pathlib import Path

# Read just metadata first
root_dir = Path(__file__).resolve().parents[1]
df, meta = pyreadstat.read_xport(root_dir / "data" / "raw" / "LLCP2014.XPT", metadataonly=True)
print("Number of columns:", len(meta.column_names))
print("Number of rows:", meta.number_rows)

target_vars = [
    'GENHLTH', '_AGEG5YR', '_BMI5CAT', 'CHECKUP1', 'INCOME2', '_RACE', 'MSCODE', 
    'FLUSHOT6', 'EMPLOY1', 'SEX', 'MARITAL', '_EDUCAG', 'SLEPTIM1', 'CVDCRHD4', 
    'HLTHCVR1', 'MENTHLTH', 'CHCKIDNY', 'USEEQUIP', '_TOTINDA', 'ADDEPEV2', 
    'RENTHOM1', 'EXERANY2', 'BLIND', 'DECIDE', 'HLTHPLN1', 'DIABETE3', '_SMOKER3'
]

present = [c for c in target_vars if c in meta.column_names]
missing = [c for c in target_vars if c not in meta.column_names]

print("Found directly:", len(present), present)
print("Missing or different name:", len(missing), missing)

if missing:
    for m in missing:
        similar = [c for c in meta.column_names if m.replace('_', '') in c or m.lower() in c.lower()]
        print(f"Candidates for {m}:", similar)
