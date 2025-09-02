# download_model.py
import os, sys, hashlib, urllib.request

MODEL_PATH = "waste_resnet50_large_ft.pt"
MODEL_URL  = "https://github.com/shauryadata/waste-classifier/releases/download/v1.0/waste_resnet50_large_ft.pt"
MODEL_SHA256 = None  # optional: paste sha256 from the release page for verification

def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, to_path: str):
    # GitHub sometimes rejects default urllib requests without a UA.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(to_path, "wb") as f:
            f.write(resp.read())

def main() -> int:
    try:
        if os.path.exists(MODEL_PATH):
            if MODEL_SHA256 and sha256_of(MODEL_PATH) != MODEL_SHA256:
                print("Checksum mismatch; re-downloading…", flush=True)
                os.remove(MODEL_PATH)
            else:
                print("Model already present.", flush=True)
                return 0

        print("Downloading model…", flush=True)
        _download(MODEL_URL, MODEL_PATH)
        print("Done:", MODEL_PATH, flush=True)

        if MODEL_SHA256:
            actual = sha256_of(MODEL_PATH)
            assert actual == MODEL_SHA256, f"Checksum mismatch: {actual}"
        return 0
    except Exception as e:
        print(f"[download_model] ERROR: {e}", flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
