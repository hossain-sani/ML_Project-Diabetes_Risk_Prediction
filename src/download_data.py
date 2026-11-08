import os
import sys
import requests
import zipfile
from project_paths import RAW_DATA_DIR, RAW_XPT_PATH, RAW_ZIP_PATH

def download_file_requests(url, output_path):
    print(f"Downloading from {url} to {output_path}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_length = r.headers.get('content-length')
        with open(output_path, 'wb') as f:
            if total_length:
                total_length = int(total_length)
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_length)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024*1024):.1f} MB / {total_length / (1024*1024):.1f} MB")
                        sys.stdout.flush()
                print()
            else:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                print("Download completed (length unknown).")

def main():
    zip_url = "https://www.cdc.gov/brfss/annual_data/2014/files/LLCP2014XPT.ZIP"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_ZIP_PATH
    xpt_path = RAW_XPT_PATH

    if not os.path.exists(xpt_path):
        if not os.path.exists(zip_path):
            download_file_requests(zip_url, zip_path)
            print(f"Downloaded {zip_path} successfully. Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"Extraction complete! {xpt_path} size: {os.path.getsize(xpt_path) / (1024*1024):.2f} MB")
    else:
        print(f"{xpt_path} already exists. Size: {os.path.getsize(xpt_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
