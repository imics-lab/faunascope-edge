"""The heartbeat: once an hour, what the device did and its state, queued for upload in the pending folder like an event (see docs/metadata.md)."""
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict

from . import device_info

FORMAT_VERSION = 1
PERIOD_SECONDS = 3600
# The camera counts as operating while its last frame is at most this many seconds old.
CAMERA_STALE_SECONDS = 2.0


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


class Heartbeat:
    """
    Accumulates the device's operating time and writes one heartbeat per period.

    tick() is called every few seconds by the main process. The time since the previous call counts as camera time if the camera delivered a frame in the last CAMERA_STALE_SECONDS, and as analyzer time if the analyzer process is alive and has loaded its models. Event and upload counts come from the processes through StatusCounters.
    """
    def __init__(self, config: Dict, status: device_info.StatusCounters):
        self.device_id = config['animal_analyzer']['device_id']
        self.pending_dir = config['animal_analyzer']['output_pending_dir']
        self.held_dir = config['animal_analyzer']['output_held_dir']
        self.status = status
        self.software = device_info.software()
        self._start_period()

    def _start_period(self) -> None:
        self.period_start = time.time()
        self.period_boot_seconds = self.last_tick = device_info.boot_seconds()
        self.operating = {'camera': 0.0, 'analyzer': 0.0}

    def tick(self, analyzer_alive: bool) -> None:
        now = device_info.boot_seconds()
        elapsed, self.last_tick = now - self.last_tick, now
        if now - self.status.camera_frame_at.value <= CAMERA_STALE_SECONDS:
            self.operating['camera'] += elapsed
        if analyzer_alive and self.status.analyzer_ready.value:
            self.operating['analyzer'] += elapsed
        if now - self.period_boot_seconds >= PERIOD_SECONDS:
            self.write()

    def write(self) -> str:
        """Write the heartbeat of the period so far to the pending folder, start the next period, and return the file's path."""
        end = time.time()
        counts = self.status.take()
        pending = os.listdir(self.pending_dir)
        heartbeat = {
            'format_version': FORMAT_VERSION,
            'device_id': self.device_id,
            'period': {'start_utc': iso_utc(self.period_start), 'end_utc': iso_utc(end)},
            'software': self.software,
            'clock': device_info.clock(end=device_info.boot_seconds()),
            'operating_seconds': {name: round(seconds) for name, seconds in self.operating.items()},
            'events': {name: counts[name] for name in ('motion', 'discarded', 'held_blank', 'kept')},
            'queue': {
                'pending': sum(1 for f in pending if f.startswith('event_') and f.endswith('.json') and f + '.failed' not in pending),
                'held_blank': sum(1 for f in os.listdir(self.held_dir) if f.startswith('event_') and f.endswith('.json')),
                'failed': sum(1 for f in pending if f.startswith('event_') and f.endswith('.json.failed')),
                'uploaded_bytes': counts['uploaded_bytes'],
            },
            'storage': {'free_bytes': shutil.disk_usage(self.pending_dir).free},
            'temperature_c': device_info.temperatures(),
        }
        path = os.path.join(self.pending_dir, f"heartbeat_{time.strftime('%Y%m%d-%H%M%S', time.localtime(end))}.json")
        with open(path + '.tmp', 'w') as f:
            json.dump(heartbeat, f, separators=(',', ':'))
        os.replace(path + '.tmp', path)
        self._start_period()
        return path
