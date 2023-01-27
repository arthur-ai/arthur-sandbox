
import os
import shutil
import time
from typing import Any, Dict, List, Optional
import requests
from datetime import date
import pandas as pd
import json
from tqdm import tqdm
import zipfile
import tempfile
from pathlib import Path


API_BASE_URL = "https://api.nasa.gov/mars-photos/api/v1"
CURIOSITY = "curiosity"
DEFAULT_DATA_DOWNLOAD_FOLDER = "./api-data-download"
DEFAULT_DATA_FOLDER = "./api-data"
MAX_RETRIES = 5

api_key = os.environ.get('NASA_API_KEY')
if api_key is None:
    raise ValueError("Please set 'NASA_API_KEY' in your environment to use the NASA API")

session = requests.Session()
session.params.update({'api_key': api_key})


class RateLimitError(Exception):
    pass


def validate_response(response):
    if response.status_code == 429:
        raise RateLimitError()
        
    elif response.status_code >= 300:
        raise ValueError(f"received response with status code {response.status_code}, body: {response.content}")
        
    return None


def download_yolo_library(skip_exists=True):
    base_folder = Path(__file__).parent.parent
    if "yolov5" in os.listdir(base_folder) and skip_exists:
        return
    else:
        print("yolov5 folder not found, downloading...")
        # download zip
        response = requests.get("https://github.com/ultralytics/yolov5/archive/refs/tags/v6.0.zip")
        validate_response(response)
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            # extract to "yolov5-6.0"
            with zipfile.ZipFile(tmp.name, 'r') as zipref:
                zipref.extractall(base_folder)

        # rename to 'yolov5'
        os.rename(base_folder / "yolov5-6.0", base_folder / "yolov5")


def nasa_get(endpoint: str, params: Optional[Dict[str, str]] = None, cur_retries=0):
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    if params is None:
        params = {}
    try:
        response = session.get(API_BASE_URL + endpoint, params=params)
        validate_response(response)
        return response

    except RateLimitError:
        # rate limit is 1000/hour so wait a bit longer than that
        print("rate limit exceeded, waiting 5 seconds")
        time.sleep(5)
        if cur_retries < MAX_RETRIES:
            return nasa_get(endpoint, params, cur_retries + 1)
        else:
            raise RateLimitError(f"Received {cur_retries} Too Many Requests responses in a row, aborting request")


def mission_manifest(rover=CURIOSITY):
    response = nasa_get(f"/rovers/{rover}")
    validate_response(response)
    return response.json()


def get_photo_response(year: int, month: int, day: int, rover=CURIOSITY, camera=None) -> List[Dict[str, Any]]:
    date = f"{year}-{month}-{day}"
    params = {'earth_date': date, 'page': 1}
    if camera is not None:
        params['camera'] = camera

    full_response = []
    first = True
    last_count = 0
    while first or last_count > 0:
        params['page'] += 1
        response = nasa_get(f"/rovers/{rover}/photos", params=params)
        try:
            validate_response(response)
        except RateLimitError:
            return response
        cur_photos = response.json()['photos']
        full_response.extend(cur_photos)
        last_count = len(cur_photos)
        first = False

    return full_response


def download_photos_in_day_range(start_date: date, end_date: date, rover=CURIOSITY, camera=None,
                                 download_folder=DEFAULT_DATA_DOWNLOAD_FOLDER,
                                 target_folder=DEFAULT_DATA_FOLDER) -> List[str]:
    date_ranges = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    print("fetching photo metadata...")
    photo_metadata = []
    for d in tqdm(date_ranges, ncols=100):
        cur_photo_response = get_photo_response(d.year, d.month, d.day, rover=rover, camera=camera)
        photo_metadata.extend(cur_photo_response)

    print(f"found {len(photo_metadata)} total photos in range, downloading new images...")
    target_folder_contents = os.listdir(target_folder)
    download_folder_contents = os.listdir(download_folder)

    to_download_metadata = []
    for pmd in photo_metadata:
        image_id = str(pmd['id'])
        target_image_folder = os.path.join(target_folder, image_id)
        download_image_folder = os.path.join(download_folder, image_id)
        download = True
        if image_id in target_folder_contents or image_id in download_folder_contents:
            download = False
            cur_image_folder_contents = os.listdir(target_image_folder)
            if not ("image.jpg" in cur_image_folder_contents and "metadata.txt" in cur_image_folder_contents):
                download = True
        else:
            os.makedirs(download_image_folder)
        if download:
            to_download_metadata.append(pmd)

    downloaded_ids = []
    for pmd in tqdm(to_download_metadata, ncols=100):
        image_id = str(pmd['id'])
        target_image_folder = os.path.join(target_folder, image_id)
        download_image_folder = os.path.join(download_folder, image_id)
        download_image_path = os.path.join(download_image_folder, "image.jpg")
        download_metadata_path = os.path.join(download_image_folder, "metadata.txt")
        response = requests.get(pmd['img_src'])
        validate_response(response)
        with open(download_image_path, 'wb') as f:
            f.write(response.content)
        with open(download_metadata_path, 'w') as f:
            f.write(json.dumps(pmd))
        shutil.copytree(download_image_folder, target_image_folder)
        downloaded_ids.append(image_id)

    print(f"successfully downloaded {len(downloaded_ids)} new images")
    return downloaded_ids
