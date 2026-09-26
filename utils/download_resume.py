import os
import sys
import time
import requests

if len(sys.argv) != 3:
    print("Usage: python utils/download_resume.py <URL> <OUTPUT>")
    sys.exit(1)

URL = sys.argv[1]
OUTPUT = sys.argv[2]

CHUNK_SIZE = 1024 * 1024

while True:
    existing = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
    print(f"\nAlready downloaded: {existing / (1024**3):.2f} GB")

    try:
        headers = {"Range": f"bytes={existing}-"}

        with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()

            if response.status_code != 206:
                print("Server did not honor Range request.")
                print("Stopping to avoid corrupting the file.")
                break

            total = int(response.headers["Content-Range"].split("/")[-1])
            print(f"Total file size: {total / (1024**3):.2f} GB")

            with open(OUTPUT, "ab") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        current = os.path.getsize(OUTPUT)
        print(f"Downloaded: {current / (1024**3):.2f} GB")

        if current >= total:
            print("\nDOWNLOAD COMPLETE!")
            break

    except Exception as e:
        current = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
        print(f"\nConnection interrupted: {type(e).__name__}: {e}")
        print(f"Saved so far: {current / (1024**3):.2f} GB")
        print("Reconnecting in 5 seconds...")
        time.sleep(5)

print(f"\nFinal file: {OUTPUT}")
if os.path.exists(OUTPUT):
    print(f"Final size: {os.path.getsize(OUTPUT) / (1024**3):.2f} GB")
