from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

RAW_XPT_PATH = RAW_DATA_DIR / "LLCP2014.XPT"
RAW_ZIP_PATH = RAW_DATA_DIR / "LLCP2014XPT.ZIP"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "diabetes_brfss2014_preprocessed.csv"
