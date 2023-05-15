import zipfile
import requests
from constants import DATA_DIR_PATH, MAESTR_ZIP_FILE_PATH, MAESTRO_DATA_URL


def download_data():
    response = requests.get(MAESTRO_DATA_URL)
    with open(str(MAESTR_ZIP_FILE_PATH), "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(str(MAESTR_ZIP_FILE_PATH), "r") as zip_ref:
        zip_ref.extractall(DATA_DIR_PATH)

    print("File downloaded and unzipped successfully!")


if __name__ == "__main__":
    download_data()
