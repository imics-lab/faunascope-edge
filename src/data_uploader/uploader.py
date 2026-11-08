import os
import time
import json
import shutil
import hashlib
import logging
from datetime import date
from typing import Dict

import requests

# Link failures pause all uploads for 1, 2, 4, ... minutes, up to this many.
MAX_BACKOFF_MINUTES = 60
STATE_FILE = '.upload_state.json'


class LinkError(Exception):
    """The server could not be reached or cannot accept uploads right now."""


class EventError(Exception):
    """The server rejected this event, or its local files cannot be read."""


class BudgetReached(Exception):
    """Today's upload budget is spent."""


def today() -> str:
    return date.today().isoformat()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def response_json(response) -> Dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


class DataUploader:
    """
    Uploads event packages (video and JSON) to the FaunaScope server over HTTPS.

    Each event goes through the server's upload API: ask how many bytes of the video the server holds, send the rest in chunks, then send the metadata with the video's size and SHA-256. The server's acceptance of the metadata confirms that the whole event arrived, and only then are the local files moved to uploaded/.

    Link failures pause all uploads with a growing delay and do not count against any event. Event failures count against that event: after event_attempts failures it is deferred for deferral_hours, and after max_deferrals deferrals it gets a .failed marker. Attempt counts, deferrals, and the bytes sent today are kept in a state file in the pending directory, so a restart neither resets the limits nor forgets them.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Data Uploader.
        :param config: The full application configuration dictionary.
        """
        self.config = config['uploader']
        self.pending_dir = config['animal_analyzer']['output_pending_dir']
        # Construct the 'uploaded' path relative to the 'pending' path
        self.uploaded_dir = os.path.join(os.path.dirname(self.pending_dir.rstrip('/')), 'uploaded')
        os.makedirs(self.uploaded_dir, exist_ok=True)

        self.server_url = str(self.config['server_url']).rstrip('/')
        self.chunk_bytes = int(self.config['chunk_size_kb']) * 1024
        self.daily_limit_bytes = int(self.config['daily_upload_limit_mb']) * 1024 * 1024
        self.timeout = self.config['request_timeout_seconds']
        self.session = requests.Session()
        self.session.headers['Authorization'] = f"Bearer {self.config['device_token']}"

        self.state_path = os.path.join(self.pending_dir, STATE_FILE)
        self.state = self._load_state()
        self.link_failures = 0
        self.resume_at = 0.0

        self.is_running = False

    def start(self) -> None:
        """Starts the main uploader loop."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_uploader.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger('DataUploader')

        if not self.config.get('enabled', True):
            logger.info("Data uploader is disabled in configuration. Files will remain in pending_upload folder.")
            return

        # A missing setting idles the uploader rather than ending its process, which would stop recording too.
        for key in ('server_url', 'device_token'):
            value = str(self.config.get(key) or '')
            if not value or value.startswith('$'):
                logger.error("uploader.%s is not set (see SERVER_URL and DEVICE_TOKEN in .env). Files will remain in pending_upload folder.", key)
                self.resume_at = float('inf')

        if self.is_running:
            logger.warning("Data uploader is already running.")
            return
        self.is_running = True
        logger.info("Starting Data Uploader for %s...", self.server_url)
        self._processing_loop()

    def stop(self) -> None:
        """Signals the main loop to terminate gracefully."""
        self.is_running = False
        print("Stopping Data Uploader...")

    def _processing_loop(self) -> None:
        """The main loop: one upload pass per scan interval, unless backing off."""
        logger = logging.getLogger('DataUploader')
        while self.is_running:
            if time.time() >= self.resume_at:
                try:
                    self._upload_pending()
                except Exception as e:
                    logger.exception("An unexpected error occurred during the upload pass: %s", e)

            for _ in range(self.config['scan_interval_seconds']):
                if not self.is_running: break
                time.sleep(1)

    def _pending_events(self):
        """JSON files waiting for upload, oldest first (event IDs start with the time)."""
        names = set(os.listdir(self.pending_dir))
        return sorted(f for f in names
                      if f.startswith('event_') and f.endswith('.json') and (f + '.failed') not in names)

    def _upload_pending(self) -> None:
        """One pass over the pending events. Stops at a link failure or when the budget is spent."""
        logger = logging.getLogger('DataUploader')
        for job_file in self._pending_events():
            if not self.is_running:
                return
            if self._bytes_today() >= self.daily_limit_bytes:
                logger.info("Daily upload budget of %d MB reached; resuming tomorrow.",
                            self.daily_limit_bytes // (1024 * 1024))
                return

            event_id = job_file[:-len('.json')]
            if self.state['events'].get(event_id, {}).get('retry_after', 0) > time.time():
                continue

            json_path = os.path.join(self.pending_dir, job_file)
            logger.info("Processing job -> %s", job_file)
            try:
                self._upload_event(event_id, json_path)
            except LinkError as e:
                self._link_failed(e)
                return
            except BudgetReached:
                logger.info("Daily upload budget reached during %s; resuming tomorrow.", event_id)
                return
            except EventError as e:
                self._event_failed(event_id, json_path, str(e))
                continue

            logger.info("Server confirmed %s. Moving local files.", event_id)
            self._move_local_files(json_path, os.path.join(self.pending_dir, event_id + '.mp4'))
            self.state['events'].pop(event_id, None)
            self._save_state()

    def _upload_event(self, event_id: str, json_path: str) -> None:
        """Bring the server to hold the complete event; returns once the server confirms it."""
        logger = logging.getLogger('DataUploader')
        video_path = os.path.join(self.pending_dir, event_id + '.mp4')
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            size = os.path.getsize(video_path)
        except (OSError, ValueError) as e:
            raise EventError(f"Cannot read the local files: {e}")

        url = f"{self.server_url}/api/v1/events/{event_id}"
        response = self._request('GET', url)
        status = response_json(response)
        if response.status_code != 200:
            raise EventError(f"Status query refused: HTTP {response.status_code} {status.get('error', '')}")
        if status.get('received'):
            logger.info("  > The server already holds %s.", event_id)
            return

        offset = status['offset']
        if offset:
            logger.info("  > Resuming at byte %d of %d.", offset, size)
        with open(video_path, 'rb') as f:
            while offset < size:
                n = min(self.chunk_bytes, size - offset)
                self._spend(n)
                f.seek(offset)
                response = self._request('PATCH', url + '/video', data=f.read(n), headers={
                    'Upload-Offset': str(offset), 'Content-Type': 'application/octet-stream'})
                body = response_json(response)
                if response.status_code == 200 or (response.status_code == 409 and 'offset' in body):
                    offset = body['offset']
                else:
                    raise EventError(f"Chunk at byte {offset} refused: HTTP {response.status_code} {body.get('error', '')}")

        response = self._request('POST', url, json={
            'metadata': metadata, 'video_size': size, 'video_sha256': file_sha256(video_path)})
        if response.status_code not in (200, 201):
            body = response_json(response)
            raise EventError(f"Metadata refused: HTTP {response.status_code} {body.get('error', '')}")

    def _request(self, method: str, url: str, **kwargs):
        """Send one request; raise LinkError when the failure lies with the link or the server."""
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as e:
            raise LinkError(f"{method} failed: {e}")
        if response.status_code in (401, 429) or response.status_code >= 500:
            raise LinkError(f"{method} answered HTTP {response.status_code} {response_json(response).get('error', '')}")
        self.link_failures = 0
        return response

    def _link_failed(self, error: LinkError) -> None:
        logger = logging.getLogger('DataUploader')
        self.link_failures += 1
        minutes = min(2 ** (self.link_failures - 1), MAX_BACKOFF_MINUTES)
        self.resume_at = time.time() + minutes * 60
        logger.warning("Link failure (%s). Pausing uploads for %d minute(s).", error, minutes)

    def _event_failed(self, event_id: str, json_path: str, error_msg: str) -> None:
        """Count a failure against the event; defer it, and eventually mark it as failed."""
        logger = logging.getLogger('DataUploader')
        record = self.state['events'].setdefault(event_id, {'attempts': 0, 'deferrals': 0})
        record['attempts'] += 1
        record['last_error'] = error_msg
        if record['attempts'] < self.config['event_attempts']:
            logger.error("  > %s (attempt %d/%d). Will retry next pass.",
                         error_msg, record['attempts'], self.config['event_attempts'])
        else:
            record['attempts'] = 0
            record['deferrals'] += 1
            if record['deferrals'] < self.config['max_deferrals']:
                record['retry_after'] = time.time() + self.config['deferral_hours'] * 3600
                logger.error("  > %s. Deferring %s for %s hours (deferral %d/%d).", error_msg, event_id,
                             self.config['deferral_hours'], record['deferrals'], self.config['max_deferrals'])
            else:
                self._fail_job(json_path, event_id, error_msg)
                self.state['events'].pop(event_id, None)
        self._save_state()

    def _bytes_today(self) -> int:
        return self.state['bytes_today'] if self.state['day'] == today() else 0

    def _spend(self, nbytes: int) -> None:
        """Count nbytes against today's budget before sending them; raise BudgetReached if they do not fit."""
        if self.state['day'] != today():
            self.state['day'], self.state['bytes_today'] = today(), 0
        if self.state['bytes_today'] + nbytes > self.daily_limit_bytes:
            raise BudgetReached()
        self.state['bytes_today'] += nbytes
        self._save_state()

    def _load_state(self) -> Dict:
        try:
            with open(self.state_path, 'r') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {'day': today(), 'bytes_today': 0, 'events': {}}

    def _save_state(self) -> None:
        with open(self.state_path + '.tmp', 'w') as f:
            json.dump(self.state, f, indent=2)
        os.replace(self.state_path + '.tmp', self.state_path)

    def _move_local_files(self, json_path: str, video_path: str) -> None:
        """
        Moves successfully uploaded files to the 'uploaded' directory.
        Both pending_upload/ and uploaded/ are bind-mounted, so shutil.move works here.
        If the move fails, the files stay in pending_upload/ — not a data loss risk: the next pass finds that the server already holds the event and tries the move again.
        """
        logger = logging.getLogger('DataUploader')
        try:
            shutil.move(json_path, os.path.join(self.uploaded_dir, os.path.basename(json_path)))
            shutil.move(video_path, os.path.join(self.uploaded_dir, os.path.basename(video_path)))
        except (IOError, OSError) as e:
            logger.error("Failed to move local files after successful upload: %s. "
                         "Files remain in pending_upload/ but upload is complete.", e)

    def _fail_job(self, json_path: str, event_id: str, error_msg: str) -> None:
        """
        Marks a job as permanently failed by creating a .failed marker file next to the
        original JSON in pending_upload/. The original files are NEVER moved or deleted.
        The scan loop skips any JSON that has a corresponding .failed marker.
        To retry, simply delete the .failed marker file.
        """
        logger = logging.getLogger('DataUploader')
        job_key = os.path.basename(json_path)
        marker_path = json_path + '.failed'
        try:
            with open(marker_path, 'w') as f:
                f.write("failed_at: %s\nevent_id: %s\nerror: %s\n" % (
                    time.strftime('%Y-%m-%d %H:%M:%S'), event_id or 'unknown', error_msg
                ))
            logger.info("  > Marked %s as permanently failed. Delete %s to retry.",
                        job_key, os.path.basename(marker_path))
        except (IOError, OSError) as e:
            logger.error("  > Could not create failure marker for %s: %s. "
                         "File will be retried on next restart.", job_key, e)
