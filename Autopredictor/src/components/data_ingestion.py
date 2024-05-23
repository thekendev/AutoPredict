#update components
import os
import urllib.request as request
import zipfile
from src.logging import logger
from src.utils.common import get_size
from src.entity.config_entity import (DataIngestionConfig)
from pathlib import Path


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.Local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename= self.config.Local_data_file
            )
            logger.info(f"{filename} Download! with the following info: \n{headers}")
        else:
            logger.info(f"file already exists of size: {get_size(Path(self.config.Local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.Local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)