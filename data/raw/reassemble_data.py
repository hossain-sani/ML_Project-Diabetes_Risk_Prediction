"""Reassemble split BRFSS ZIP parts and extract the XPT dataset."""

import argparse
import shutil
import zipfile
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent
PARTS = (
    RAW_DIR / "LLCP2014XPT001.ZIP",
    RAW_DIR / "LLCP2014XPT002.ZIP",
)
REASSEMBLED_ZIP = RAW_DIR / "LLCP2014XPT_reassembled.ZIP"
XPT_FILE = RAW_DIR / "LLCP2014.XPT"


def reassemble_and_extract(force=False):
    missing_parts = [part for part in PARTS if not part.exists()]
    if missing_parts:
        names = ", ".join(part.name for part in missing_parts)
        raise FileNotFoundError(f"Missing split archive part(s): {names}")

    if XPT_FILE.exists() and not force:
        print(f"{XPT_FILE.name} already exists. Use --force to recreate it.")
        return

    print(f"Joining {PARTS[0].name} and {PARTS[1].name}...")
    with REASSEMBLED_ZIP.open("wb") as output:
        for part in PARTS:
            with part.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)

    print(f"Checking {REASSEMBLED_ZIP.name}...")
    with zipfile.ZipFile(REASSEMBLED_ZIP) as archive:
        archive.testzip()
        members = archive.namelist()
        xpt_members = [member for member in members if member.strip().upper().endswith(".XPT")]
        if not xpt_members:
            raise ValueError("The reassembled archive does not contain an XPT file.")
        with archive.open(xpt_members[0]) as input_file, XPT_FILE.open("wb") as output:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)

    print(f"Created {XPT_FILE.name} ({XPT_FILE.stat().st_size / (1024 * 1024):.2f} MB).")
    print(f"Reassembled archive kept at {REASSEMBLED_ZIP.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join the split BRFSS ZIP parts and extract LLCP2014.XPT."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the XPT file even when it already exists.",
    )
    args = parser.parse_args()
    reassemble_and_extract(force=args.force)
