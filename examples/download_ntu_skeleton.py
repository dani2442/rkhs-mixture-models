"""Download and extract NTU RGB+D skeleton data."""
from pathlib import Path
import zipfile
import gdown

ROOT = Path("data/ntu_rgbd_skeleton")
ROOT.mkdir(parents=True, exist_ok=True)

# Official skeleton-only file IDs referenced by the dataset authors
FILE_IDS = {
    "nturgbd_skeletons_s001_to_s017.zip": "1CUZnBtYwifVXS21yVg62T-vrPVayso5H",  # NTU60 skeletons
    "nturgbd_skeletons_s018_to_s032.zip": "1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w",  # extra setups for NTU120
}

def download_and_unzip(name: str, file_id: str, out_dir: Path) -> Path:
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = out_dir / name

    if not zip_path.exists():
        print(f"Downloading {name} ...")
        gdown.download(url, str(zip_path), quiet=False)

    extract_dir = out_dir / name.replace(".zip", "")
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    return extract_dir

dirs = {}
for fname, fid in FILE_IDS.items():
    dirs[fname] = download_and_unzip(fname, fid, ROOT)

print("Done. Extracted folders:")
for k, v in dirs.items():
    print(" ", k, "->", v)
