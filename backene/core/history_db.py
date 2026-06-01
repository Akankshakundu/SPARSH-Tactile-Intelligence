"""
Sparsh Local History Database.
Stores uploaded Braille images, HUD annotations, and text results inside the
backend/uploads directory with a JSON index for full retrieval and audit.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
from typing import Optional

# Base paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
DB_FILE = os.path.join(UPLOADS_DIR, "database.json")

# Ensure the upload directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)


def init_db():
    """Initializes the offline database JSON file if missing."""
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump([], f)


def save_upload_record(
    original_img: np.ndarray,
    annotated_img: Optional[np.ndarray],
    text: str,
    confidence: float,
    cell_count: int,
    dot_count: int,
    processing_time_ms: float
) -> dict:
    """
    Saves the uploaded original and annotated images to disk and logs
    the metadata in the local JSON database.
    """
    init_db()
    
    timestamp_id = int(time.time())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filenames
    orig_filename = f"{file_prefix}_{timestamp_id}_orig.jpg"
    ann_filename = f"{file_prefix}_{timestamp_id}_ann.jpg"
    
    orig_path = os.path.join(UPLOADS_DIR, orig_filename)
    ann_path = os.path.join(UPLOADS_DIR, ann_filename)
    
    # Save files to disk
    cv2.imwrite(orig_path, original_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if annotated_img is not None:
        cv2.imwrite(ann_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        ann_filename = None
        
    # Read existing entries
    try:
        with open(DB_FILE, "r") as f:
            records = json.load(f)
    except Exception:
        records = []
        
    # Create record object
    new_record = {
        "id": str(timestamp_id),
        "timestamp": date_str,
        "original_image": f"/uploads/{orig_filename}",
        "annotated_image": f"/uploads/{ann_filename}" if ann_filename else None,
        "text": text if text.strip() else "[No cells recognized]",
        "confidence": confidence,
        "cell_count": cell_count,
        "dot_count": dot_count,
        "processing_time": processing_time_ms
    }
    
    # Prepend for newest-first order
    records.insert(0, new_record)
    
    # Cap at 50 records to save disk space
    if len(records) > 50:
        # Delete old files from disk
        old_record = records.pop()
        try:
            o_file = os.path.join(UPLOADS_DIR, os.path.basename(old_record["original_image"]))
            if os.path.exists(o_file):
                os.remove(o_file)
            if old_record["annotated_image"]:
                a_file = os.path.join(UPLOADS_DIR, os.path.basename(old_record["annotated_image"]))
                if os.path.exists(a_file):
                    os.remove(a_file)
        except Exception as e:
            print(f"[DB] Error cleanup: {e}")
            
    # Save back to database.json
    try:
        with open(DB_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[DB] Failed to write database: {e}")
        
    return new_record


def get_all_records() -> list[dict]:
    """Retrieves all logged history records."""
    init_db()
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def clear_all_history():
    """Deletes all history database logs and saved images."""
    try:
        if os.path.exists(UPLOADS_DIR):
            for file in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        with open(DB_FILE, "w") as f:
            json.dump([], f)
        return True
    except Exception as e:
        print(f"[DB] Failed to clear history: {e}")
        return False
