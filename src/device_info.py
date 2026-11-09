"""Facts about the device that event metadata and heartbeats report: its software, its clock, and its hardware's state, plus the counters its processes share for the heartbeat."""
import ctypes
import glob
import hashlib
import multiprocessing
import os
import time
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# adjtimex() returns TIME_ERROR while the kernel considers the clock unsynchronized.
TIME_ERROR = 5


def source_sha256() -> str:
    """SHA-256 over main.py and the Python files under src/, each with its path, in a fixed order."""
    digest = hashlib.sha256()
    paths = [os.path.join(ROOT, 'main.py')] + sorted(glob.glob(os.path.join(ROOT, 'src', '**', '*.py'), recursive=True))
    for path in paths:
        digest.update(os.path.relpath(path, ROOT).encode() + b'\0')
        with open(path, 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    """The commit the code comes from: FAUNASCOPE_COMMIT, which the Docker image is built with, or else the checkout's HEAD."""
    commit = os.environ.get('FAUNASCOPE_COMMIT')
    if commit:
        return commit
    git_dir = os.path.join(ROOT, '.git')
    try:
        with open(os.path.join(git_dir, 'HEAD')) as f:
            head = f.read().strip()
        if head.startswith('ref: '):
            ref = head[len('ref: '):]
            if os.path.exists(os.path.join(git_dir, ref)):
                with open(os.path.join(git_dir, ref)) as f:
                    head = f.read().strip()
            else:
                with open(os.path.join(git_dir, 'packed-refs')) as f:
                    head = next(line.split()[0] for line in f if line.rstrip().endswith(' ' + ref))
        return head[:7]
    except (OSError, StopIteration):
        return None


def software() -> Dict:
    commit = git_commit()
    return {'source_sha256': source_sha256(), **({'commit': commit} if commit else {})}


def clock_synchronized() -> Optional[bool]:
    """Whether the kernel considers the system clock synchronized (by NTP), or None if it cannot be asked."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # A zeroed struct timex (208 bytes on 64-bit Linux): modes 0 only reads the clock's state.
        state = libc.adjtimex(ctypes.create_string_buffer(512))
    except (OSError, AttributeError):
        return None
    return None if state < 0 else state != TIME_ERROR


def boot_id() -> Optional[str]:
    try:
        with open('/proc/sys/kernel/random/boot_id') as f:
            return f.read().strip()
    except OSError:
        return None


def boot_seconds() -> float:
    """Seconds since boot, including time suspended; unaffected by changes to the system clock."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def clock(**boot_seconds_at) -> Dict:
    """The clock's state, with boot-time readings named by the caller, e.g. clock(boot_seconds_at_start=...)."""
    return {'synchronized': clock_synchronized(), 'boot_id': boot_id(),
            **{f'boot_seconds_at_{name}': round(value, 3) for name, value in boot_seconds_at.items()}}


def temperatures() -> Dict[str, float]:
    """Each thermal zone's temperature in degrees Celsius, by the zone's type ('cpu-thermal' as 'cpu')."""
    result = {}
    for zone in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        try:
            with open(os.path.join(zone, 'type')) as f:
                name = f.read().strip()
            with open(os.path.join(zone, 'temp')) as f:
                millidegrees = int(f.read())
        except (OSError, ValueError):
            continue
        if name.endswith('-thermal'):
            name = name[:-len('-thermal')]
        result[name] = round(millidegrees / 1000, 1)
    return result


class StatusCounters:
    """
    What the device's processes report for the heartbeat, in shared memory: counts of events by outcome and of bytes uploaded, the boot time of the camera's last frame, and whether the analyzer has loaded its models.

    An instance is created before the processes start and handed to each of them.
    """
    COUNTS = ('motion', 'discarded', 'held_blank', 'kept', 'uploaded_bytes')

    def __init__(self):
        self.counts = {name: multiprocessing.Value('q', 0) for name in self.COUNTS}
        self.camera_frame_at = multiprocessing.Value('d', 0.0)
        self.analyzer_ready = multiprocessing.Value('b', 0)

    def add(self, name: str, n: int = 1) -> None:
        with self.counts[name].get_lock():
            self.counts[name].value += n

    def take(self) -> Dict[str, int]:
        """The counts since the last call, which resets them."""
        result = {}
        for name, value in self.counts.items():
            with value.get_lock():
                result[name], value.value = value.value, 0
        return result
