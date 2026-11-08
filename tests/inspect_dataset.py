import pandas as pd
from pathlib import Path

df = pd.read_sas(Path(__file__).resolve().parents[1] / "data" / "raw" / "LLCP2014.XPT", format="xport")
print(df.shape)  # expect about (464644, 279)
