import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path("../data")
TEST_DIR = DATA_DIR / "test"
DEBLUR_DIR = DATA_DIR / "deblur"
REALESRGAN_DIR = DATA_DIR / "real-esrgan"
FINAL_DIR = DATA_DIR / "real-blur"

BASE_DIR = Path(__file__).resolve().parent
MDEBLUR_DIR = BASE_DIR.parent / "deblurganv2"
MREALESRGAN_DIR = BASE_DIR.parent / "real-esrgan"

DEBLUR_SCRIPT = "../deblurganv2/predict.py"
REALESRGAN_SCRIPT = "../real-esrgan/predict.py"

BACKUP_TEST_DIR = DATA_DIR / "test_backup_original"


def clear_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


print("Running Deblur...")
subprocess.run(["py", DEBLUR_SCRIPT],cwd=MDEBLUR_DIR, check=True)

if BACKUP_TEST_DIR.exists():
    shutil.rmtree(BACKUP_TEST_DIR)

shutil.copytree(TEST_DIR, BACKUP_TEST_DIR)

clear_folder(TEST_DIR)

for img_path in DEBLUR_DIR.iterdir():
    if img_path.is_file():
        shutil.copy2(img_path, TEST_DIR / img_path.name)

print("Running Real-ESRGAN...")
subprocess.run(["py", REALESRGAN_SCRIPT],cwd=MREALESRGAN_DIR, check=True)


FINAL_DIR.mkdir(parents=True, exist_ok=True)
if FINAL_DIR.exists():
    shutil.rmtree(FINAL_DIR)

shutil.copytree(REALESRGAN_DIR, FINAL_DIR)

shutil.rmtree(TEST_DIR)
shutil.copytree(BACKUP_TEST_DIR, TEST_DIR)

print("Done!")
print(f"Final output saved to: {FINAL_DIR}")