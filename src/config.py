import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    LAW_DB_PATH = Path(os.getenv("LAW_DB_PATH", BASE_DIR / "data" / "db" / "laws.db"))
    CRIME_DB_PATH = Path(os.getenv("CRIME_DB_PATH", BASE_DIR / "data" / "db" / "crime.db"))

config = Config()
