"""The uploader against a fake server that speaks the FaunaScope upload API.

Heartbeats go to /api/v1/heartbeats, which the fake server logs with "heartbeats" in place of the event ID.

The fake server keeps uploads in memory and can be told to fail particular requests: drop the connection without answering, answer with an error status, refuse a chunk for its offset, or store a chunk or an event and then drop the confirmation.
"""
import gzip
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.data_uploader import uploader as uploader_module
from src.data_uploader.uploader import DataUploader

TOKEN = "test-token"
KB = 1024


class FakeServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.partial = {}    # event_id -> bytes received
        self.stored = {}     # event_id -> (video, metadata)
        self.faults = []     # (method, action) or (method, action, event_id), each used by the first matching request
        self.log = []        # (method, event_id, Upload-Offset or None, status)
        self.metadata_bytes = {}  # event_id -> size of the metadata request as sent
        self.heartbeats = []      # heartbeats received

    def take_fault(self, method, event_id):
        for i, fault in enumerate(self.faults):
            if fault[0] == method and fault[2:] in ((), (event_id,)):
                return self.faults.pop(i)[1]
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_one(self, method):
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        parts = self.path.split("/")
        event_id = parts[4] if len(parts) > 4 else parts[3]
        offset = self.headers.get("Upload-Offset")
        fault = server.take_fault(method, event_id)
        if fault == "drop":
            server.log.append((method, event_id, offset, "dropped"))
            self.close_connection = True
            self.connection.shutdown(2)
            return
        if fault == "wrong-offset":
            held = len(server.partial.get(event_id, b""))
            server.log.append((method, event_id, offset, 409))
            return self.reply(409, {"error": "offset mismatch", "offset": held})
        if isinstance(fault, int):
            server.log.append((method, event_id, offset, fault))
            return self.reply(fault, {"error": "injected"})
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self.reply(401, {"error": "Missing or invalid device token."})

        status, answer = self.answer(method, event_id, offset, body)
        server.log.append((method, event_id, offset, status))
        if fault == "store-then-drop":
            self.close_connection = True
            self.connection.shutdown(2)
            return
        self.reply(status, answer)

    def answer(self, method, event_id, offset, body):
        server = self.server
        held = server.partial.get(event_id, b"")
        if method == "GET":
            if event_id in server.stored:
                return 200, {"received": True}
            return 200, {"received": False, "offset": len(held)}
        if method == "PATCH":
            if int(offset) != len(held):
                return 409, {"error": "offset mismatch", "offset": len(held)}
            server.partial[event_id] = held + body
            return 200, {"offset": len(held) + len(body)}
        sent = len(body)
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        if event_id == "heartbeats":
            server.heartbeats.append(json.loads(body))
            return 201, {"received": True}
        server.metadata_bytes[event_id] = sent
        request = json.loads(body)
        if event_id in server.stored:
            return 200, {"received": True}
        if len(held) < request["video_size"]:
            return 409, {"error": "incomplete", "offset": len(held)}
        if hashlib.sha256(held).hexdigest() != request["video_sha256"]:
            server.partial.pop(event_id)
            return 422, {"error": "checksum mismatch", "offset": 0}
        server.stored[event_id] = (server.partial.pop(event_id), request["metadata"])
        return 201, {"received": True}

    def do_GET(self):
        self.handle_one("GET")

    def do_PATCH(self):
        self.handle_one("PATCH")

    def do_POST(self):
        self.handle_one("POST")


@pytest.fixture
def server():
    server = FakeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def make_uploader(config, server, monkeypatch):
    """Build an uploader for the fake server; keyword arguments override uploader settings. Requests are not paced unless a test asks for it."""
    monkeypatch.setattr(uploader_module, "MIN_REQUEST_INTERVAL_SECONDS", 0)

    def make(**settings):
        config["uploader"] = {
            "enabled": True,
            "scan_interval_seconds": 1,
            "server_url": f"http://127.0.0.1:{server.server_address[1]}",
            "device_token": TOKEN,
            "chunk_size_kb": 64,
            "daily_upload_limit_mb": 800,
            "event_attempts": 3,
            "deferral_hours": 24,
            "max_deferrals": 3,
            "request_timeout_seconds": 5,
            **settings,
        }
        os.makedirs(config["animal_analyzer"]["output_pending_dir"], exist_ok=True)
        up = DataUploader(config)
        up.is_running = True
        return up
    return make


def add_event(pending_dir, event_id, size):
    """Put a video of size random bytes and its metadata into the pending directory."""
    video = os.urandom(size)
    with open(os.path.join(pending_dir, f"{event_id}.mp4"), "wb") as f:
        f.write(video)
    metadata = {"format_version": 2, "event_id": event_id, "device_id": "test-device"}
    with open(os.path.join(pending_dir, f"{event_id}.json"), "w") as f:
        json.dump(metadata, f)
    return video, metadata


EVENT = "event_20261108-160000_0000beef"


def patches(server):
    return [(int(offset), status) for method, _, offset, status in server.log if method == "PATCH"]


def test_an_event_is_uploaded_in_chunks_and_moved_after_confirmation(make_uploader, server):
    up = make_uploader()
    video, metadata = add_event(up.pending_dir, EVENT, 200 * KB)

    up._upload_pending()

    assert server.stored[EVENT] == (video, metadata)
    assert [offset for offset, _ in patches(server)] == [0, 64 * KB, 128 * KB, 192 * KB]
    assert sorted(os.listdir(up.uploaded_dir)) == [f"{EVENT}.json", f"{EVENT}.mp4"]
    assert not os.path.exists(os.path.join(up.pending_dir, f"{EVENT}.mp4"))
    assert up.state["bytes_today"] == 200 * KB


def test_events_go_oldest_first(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, "event_20261108-170000_00000002", KB)
    add_event(up.pending_dir, "event_20261107-090000_00000001", KB)

    up._upload_pending()

    posts = [event_id for method, event_id, _, _ in server.log if method == "POST"]
    assert posts == ["event_20261107-090000_00000001", "event_20261108-170000_00000002"]


def test_an_interrupted_upload_resumes_without_resending_confirmed_bytes(make_uploader, server):
    up = make_uploader()
    video, _ = add_event(up.pending_dir, EVENT, 200 * KB)
    server.faults = [("PATCH", None), ("PATCH", "drop")]

    up._upload_pending()
    assert EVENT not in server.stored
    assert up.resume_at > time.time()
    assert EVENT not in up.state["events"]  # a link failure does not count against the event

    up.resume_at = 0
    up._upload_pending()

    assert server.stored[EVENT][0] == video
    assert patches(server) == [(0, 200), (64 * KB, "dropped"), (64 * KB, 200), (128 * KB, 200), (192 * KB, 200)]


def test_a_lost_chunk_confirmation_resumes_after_the_stored_chunk(make_uploader, server):
    up = make_uploader()
    video, _ = add_event(up.pending_dir, EVENT, 200 * KB)
    server.faults = [("PATCH", None), ("PATCH", "store-then-drop")]

    up._upload_pending()
    up.resume_at = 0
    up._upload_pending()

    assert server.stored[EVENT][0] == video
    assert [offset for offset, _ in patches(server)] == [0, 64 * KB, 128 * KB, 192 * KB]


def test_a_chunk_refused_for_its_offset_is_resent_at_the_offset_the_server_names(make_uploader, server):
    up = make_uploader()
    video, _ = add_event(up.pending_dir, EVENT, 200 * KB)
    server.faults = [("PATCH", None), ("PATCH", "wrong-offset")]

    up._upload_pending()

    assert server.stored[EVENT][0] == video
    assert patches(server) == [(0, 200), (64 * KB, 409), (64 * KB, 200), (128 * KB, 200), (192 * KB, 200)]


def test_link_failures_back_off_exponentially_and_never_count_against_the_event(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    server.faults = [("GET", 503), ("GET", 429), ("GET", "drop")]

    delays = []
    for _ in range(3):
        before = time.time()
        up._upload_pending()
        delays.append(round((up.resume_at - before) / 60))
        up.resume_at = 0

    assert delays == [1, 2, 4]
    assert up.state["events"] == {}
    up._upload_pending()
    assert EVENT in server.stored
    assert up.link_failures == 0


def test_the_backoff_is_capped_at_an_hour(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    up.link_failures = 10
    server.faults = [("GET", 500)]

    before = time.time()
    up._upload_pending()

    assert round((up.resume_at - before) / 60) == 60


def test_a_wrong_token_is_a_link_failure(make_uploader, server):
    up = make_uploader(device_token="wrong")
    add_event(up.pending_dir, EVENT, KB)

    up._upload_pending()

    assert up.resume_at > time.time()
    assert up.state["events"] == {}
    assert EVENT not in server.stored


def test_a_rejected_event_is_deferred_after_three_attempts_and_marked_failed_after_three_deferrals(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    other = "event_20261108-170000_00000002"
    add_event(up.pending_dir, other, KB)
    server.faults = [("POST", 422, EVENT)] * 9

    for deferral in (1, 2):
        for attempt in (1, 2, 3):
            up._upload_pending()
        record = up.state["events"][EVENT]
        assert record["deferrals"] == deferral and record["attempts"] == 0
        assert record["retry_after"] > time.time() + 23 * 3600
        up._upload_pending()  # deferred: not tried
        assert sum(1 for m, e, _, _ in server.log if m == "POST" and e == EVENT) == 3 * deferral
        record["retry_after"] = 0

    for attempt in (1, 2, 3):
        up._upload_pending()

    assert os.path.exists(os.path.join(up.pending_dir, f"{EVENT}.json.failed"))
    assert os.path.exists(os.path.join(up.pending_dir, f"{EVENT}.mp4"))  # files are never deleted
    assert EVENT not in up.state["events"]
    assert other in server.stored  # one bad event does not hold up the others


def test_missing_local_video_counts_against_the_event_without_contacting_the_server(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    os.remove(os.path.join(up.pending_dir, f"{EVENT}.mp4"))

    up._upload_pending()

    assert up.state["events"][EVENT]["attempts"] == 1
    assert server.log == []


def test_the_daily_budget_stops_uploads_and_they_resume_the_next_day(make_uploader, server, monkeypatch):
    up = make_uploader(chunk_size_kb=256, daily_upload_limit_mb=1)
    video, _ = add_event(up.pending_dir, EVENT, 1536 * KB)
    monkeypatch.setattr(uploader_module, "today", lambda: "2026-11-08")

    up._upload_pending()
    up._upload_pending()

    assert len(server.partial[EVENT]) == 1024 * KB
    assert up.state["bytes_today"] == 1024 * KB
    assert os.path.exists(os.path.join(up.pending_dir, f"{EVENT}.mp4"))

    monkeypatch.setattr(uploader_module, "today", lambda: "2026-11-09")
    up._upload_pending()

    assert server.stored[EVENT][0] == video
    assert up.state == {"day": "2026-11-09", "bytes_today": 512 * KB, "events": {}}


def test_a_lost_confirmation_is_recovered_on_the_next_pass(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, 100 * KB)
    server.faults = [("POST", "store-then-drop")]

    up._upload_pending()
    assert EVENT in server.stored
    assert os.path.exists(os.path.join(up.pending_dir, f"{EVENT}.mp4"))

    up.resume_at = 0
    up._upload_pending()

    assert sorted(os.listdir(up.uploaded_dir)) == [f"{EVENT}.json", f"{EVENT}.mp4"]
    assert len(patches(server)) == 2  # nothing was sent twice


def test_state_survives_a_restart(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    server.faults = [("POST", 422)]
    up._upload_pending()

    restarted = make_uploader()

    assert restarted.state["events"][EVENT]["attempts"] == 1
    assert restarted.state["bytes_today"] == KB


def test_failed_markers_and_the_state_file_are_not_uploaded(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    open(os.path.join(up.pending_dir, f"{EVENT}.json.failed"), "w").close()
    up._save_state()

    up._upload_pending()

    assert server.log == []


def test_the_metadata_is_sent_gzipped(make_uploader, server):
    up = make_uploader()
    _, metadata = add_event(up.pending_dir, EVENT, 10 * KB)
    metadata["tracks"] = [{"track_id": t, "class": "animal", "boxes": {
        "frame": list(range(3000)), "xyxy": [[280, 210, 360, 270]] * 3000,
        "detected": [i % 5 == 0 for i in range(3000)], "confidence": [0.91 if i % 5 == 0 else None for i in range(3000)]}} for t in range(7)]
    with open(os.path.join(up.pending_dir, EVENT + ".json"), "w") as f:
        json.dump(metadata, f)

    up._upload_pending()

    assert server.stored[EVENT][1] == metadata
    assert server.metadata_bytes[EVENT] < len(json.dumps(metadata)) / 10


def test_requests_are_paced_to_the_servers_rate_limit(make_uploader, server, monkeypatch):
    up = make_uploader()
    monkeypatch.setattr(uploader_module, "MIN_REQUEST_INTERVAL_SECONDS", 0.1)
    add_event(up.pending_dir, EVENT, 200 * KB)

    started = time.monotonic()
    up._upload_pending()

    # GET, four chunks, and POST: five intervals between six requests.
    assert len(server.log) == 6 and time.monotonic() - started >= 0.5


def add_heartbeat(pending_dir, name="heartbeat_20261108-170000.json"):
    heartbeat = {"format_version": 1, "device_id": "test-device", "events": {"motion": 3}}
    with open(os.path.join(pending_dir, name), "w") as f:
        json.dump(heartbeat, f)
    return heartbeat


def test_heartbeats_are_sent_before_events_and_deleted_once_confirmed(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    first = add_heartbeat(up.pending_dir, "heartbeat_20261108-160000.json")
    second = add_heartbeat(up.pending_dir, "heartbeat_20261108-170000.json")

    up._upload_pending()

    assert server.heartbeats == [first, second]
    assert [event_id for method, event_id, _, _ in server.log if method == "POST"] == ["heartbeats", "heartbeats", EVENT]
    assert not any(f.startswith("heartbeat_") for f in os.listdir(up.pending_dir))


def test_a_link_failure_on_a_heartbeat_stops_the_pass(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    add_heartbeat(up.pending_dir)
    server.faults = [("POST", "drop", "heartbeats")]

    up._upload_pending()

    assert EVENT not in server.stored and up.resume_at > time.time()
    assert "heartbeat_20261108-170000.json" in os.listdir(up.pending_dir)


def test_a_refused_heartbeat_is_set_aside(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, KB)
    add_heartbeat(up.pending_dir)
    server.faults = [("POST", 422, "heartbeats")]

    up._upload_pending()

    assert EVENT in server.stored
    assert "heartbeat_20261108-170000.json.failed" in os.listdir(up.pending_dir)
    up._upload_pending()
    assert [method for method, event_id, _, _ in server.log if event_id == "heartbeats"] == ["POST"]


def test_the_bytes_sent_are_counted_for_the_heartbeat(make_uploader, server):
    up = make_uploader()
    add_event(up.pending_dir, EVENT, 200 * KB)

    up._upload_pending()

    assert up.status.take()["uploaded_bytes"] > 200 * KB
