"""The heartbeat's operating time, counts, and file, with a scripted boot clock."""
import json
import os

import pytest

from src import device_info
from src.device_info import StatusCounters
from src.heartbeat import Heartbeat


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(device_info, "boot_seconds", lambda: now["t"])
    return now


@pytest.fixture
def heartbeat(config, clock):
    for key in ("output_pending_dir", "output_held_dir"):
        os.makedirs(config["animal_analyzer"][key])
    return Heartbeat(config, StatusCounters())


def test_operating_time_counts_while_the_camera_delivers_and_the_analyzer_is_ready(heartbeat, clock):
    status = heartbeat.status
    status.analyzer_ready.value = 1
    status.camera_frame_at.value = clock["t"] = 105.0
    heartbeat.tick(analyzer_alive=True)
    clock["t"] = 110.0          # no frame for 5 seconds
    heartbeat.tick(analyzer_alive=True)
    clock["t"] = 115.0
    heartbeat.tick(analyzer_alive=False)
    assert heartbeat.operating == {"camera": 5.0, "analyzer": 10.0}


def test_a_heartbeat_is_written_once_the_period_is_over(heartbeat, clock, config):
    pending = config["animal_analyzer"]["output_pending_dir"]
    held = config["animal_analyzer"]["output_held_dir"]
    for name in ("event_1.json", "event_1.mp4", "event_2.json", "event_2.json.failed"):
        open(os.path.join(pending, name), "w").close()
    open(os.path.join(held, "event_3.json"), "w").close()
    heartbeat.status.add("motion", 4)
    heartbeat.status.add("kept", 3)
    heartbeat.status.add("uploaded_bytes", 1000)

    clock["t"] = 3699.0
    heartbeat.tick(analyzer_alive=True)
    assert not any(f.startswith("heartbeat_") for f in os.listdir(pending))
    clock["t"] = 3700.0
    heartbeat.tick(analyzer_alive=True)

    [name] = [f for f in os.listdir(pending) if f.startswith("heartbeat_")]
    with open(os.path.join(pending, name)) as f:
        data = json.load(f)
    assert data["format_version"] == 1 and data["device_id"] == "test-device"
    assert data["clock"]["boot_seconds_at_end"] == 3700.0
    assert data["events"] == {"motion": 4, "discarded": 0, "held_blank": 0, "kept": 3}
    assert data["queue"] == {"pending": 1, "held_blank": 1, "failed": 1, "uploaded_bytes": 1000}
    assert data["storage"]["free_bytes"] > 0
    assert set(data["period"]) == {"start_utc", "end_utc"} and len(data["software"]["source_sha256"]) == 64

    # The next period starts from zero.
    assert heartbeat.operating == {"camera": 0.0, "analyzer": 0.0}
    assert heartbeat.status.take()["kept"] == 0


def test_the_commit_comes_from_the_build(monkeypatch):
    monkeypatch.setenv("FAUNASCOPE_COMMIT", "abc1234")
    assert device_info.software()["commit"] == "abc1234"


def test_the_clock_reports_its_state_and_the_boot_time():
    clock = device_info.clock(start=12.34567)
    assert clock["synchronized"] in (True, False, None)
    assert clock["boot_seconds_at_start"] == 12.346
    assert set(clock) == {"synchronized", "boot_id", "boot_seconds_at_start"}
