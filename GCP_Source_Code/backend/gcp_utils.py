"""
Wasel v4 Pro: GCP Utilities
Handles GCS persistence and cloud-native logging.
All GCP imports are conditional — app works 100% without them.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Conditional GCP Imports ───
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None


def is_running_on_gcp():
    """Detect if the app is running in a Cloud Run environment."""
    return os.getenv("K_SERVICE") is not None


def setup_cloud_logging():
    """Setup Google Cloud Logging if running on GCP."""
    if is_running_on_gcp():
        try:
            import google.cloud.logging
            client = google.cloud.logging.Client()
            client.setup_logging()
            logger.info("✅ Cloud Logging initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Cloud Logging: {e}")


class GCSManager:
    """Manages synchronization between local data and GCS buckets."""

    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or os.getenv("WASEL_GCS_BUCKET")
        self.client = None

        if self.bucket_name and GCS_AVAILABLE:
            try:
                self.client = storage.Client()
                logger.info(f"☁️ GCS client initialized for bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"⚠️ GCS client init failed (running locally?): {e}")
                self.client = None

    def upload_file(self, local_path: str, remote_path: str):
        """Upload a file to GCS."""
        if not self.client:
            return
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"☁️ Uploaded {remote_path}")
        except Exception as e:
            logger.error(f"❌ GCS Upload error: {e}")

    def download_file(self, remote_path: str, local_path: str):
        """Download a file from GCS."""
        if not self.client:
            return False
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            if blob.exists():
                blob.download_to_filename(local_path)
                return True
        except Exception as e:
            logger.error(f"❌ GCS Download error: {e}")
        return False

    def sync_directory(self, local_dir: Path, prefix: str):
        """Sync a local directory with GCS (one-way download on startup)."""
        if not self.client:
            return
        try:
            bucket = self.client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            count = 0
            for blob in blobs:
                rel_path = os.path.relpath(blob.name, prefix)
                if rel_path == ".":
                    continue
                dest_path = local_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if not dest_path.exists():
                    blob.download_to_filename(str(dest_path))
                    count += 1
            if count > 0:
                logger.info(f"📥 Downloaded {count} files from gs://{self.bucket_name}/{prefix}")
        except Exception as e:
            logger.error(f"❌ GCS Sync error: {e}")
