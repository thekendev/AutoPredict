#Update the entity for data ingestion
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    Local_data_file: Path
    unzip_dir: Path




@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path