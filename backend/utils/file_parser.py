"""
File Parsing and Ingestion Utility
Supports CSV, XLSX, and XLS formats with validation and size guards.
Independent of web framework for universal compatibility.
"""
import io
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, BinaryIO
import pandas as pd

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safeguard


def parse_and_store_file(filename: str, file_bytes: bytes) -> Tuple[str, pd.DataFrame, str]:
    """
    Validates uploaded file content, saves to disk, and parses to DataFrame.
    Returns: (dataset_id, dataframe, filename)
    """
    clean_filename = filename or "uploaded_dataset.csv"
    ext = Path(clean_filename).suffix.lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: .csv, .xlsx, .xls")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds maximum allowed size of 50 MB ({len(file_bytes)/(1024*1024):.1f} MB uploaded)."
        )

    dataset_id = str(uuid.uuid4())[:8]
    saved_path = UPLOAD_DIR / f"{dataset_id}_{clean_filename}"

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unknown extension {ext}")

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    return dataset_id, df, clean_filename
