
from pathlib import Path
import os
import requests
import tempfile
import zipfile


def download_yolo_library(skip_exists=True):
    base_folder = Path(__file__).parent.parent
    if "yolov5" in os.listdir(base_folder) and skip_exists:
        return
    else:
        print("yolov5 folder not found, downloading...")
        # download zip
        response = requests.get("https://github.com/ultralytics/yolov5/archive/refs/tags/v6.0.zip")
        assert response.status_code == 200
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            # extract to "yolov5-6.0"
            with zipfile.ZipFile(tmp.name, 'r') as zipref:
                zipref.extractall(base_folder)

        # rename to 'yolov5'
        os.rename(base_folder / "yolov5-6.0", base_folder / "yolov5")
