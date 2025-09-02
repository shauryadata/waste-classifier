# download_model.py
import os, urllib.request, sys, hashlib

MODEL_PATH = "waste_resnet50_large_ft.pt"
# paste the asset URL here:
MODEL_URL  = "https://github.com/<your-username>/waste-classifier/releases/download/v1.0/waste_resnet50_large_ft.pt"
MODEL_SHA256 = None  

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if os.path.exists(MODEL_PATH):
        if MODEL_SHA256 and sha256_of(MODEL_PATH) != MODEL_SHA256:
            print("Checksum mismatch; re-downloading…")
            os.remove(MODEL_PATH)
        else:
            print("Model already present.")
            return 0
    print("Downloading model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done:", MODEL_PATH)
    if MODEL_SHA256:
        assert sha256_of(MODEL_PATH) == MODEL_SHA256, "Checksum mismatch"
    return 0

if __name__ == "__main__":
    sys.exit(main())