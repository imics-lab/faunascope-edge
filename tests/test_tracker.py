import pytest

from src.animal_analyzer.analyzer import AnimalAnalyzer


@pytest.fixture
def analyzer(config):
    return AnimalAnalyzer(None, config)


def detection(x, y, label="Fox", confidence=0.9):
    return {"label": label, "confidence": confidence, "centroid": (x, y), "box_xywh": (x, y, 20, 20)}


def test_new_detections_get_new_ids(analyzer):
    tracks, next_id = analyzer._update_tracker({}, [detection(100, 100), detection(400, 400)], 0)
    assert next_id == 2
    assert {t["centroid"] for t in tracks.values()} == {(100, 100), (400, 400)}


def test_a_nearby_detection_continues_its_track(analyzer):
    tracks, next_id = analyzer._update_tracker({}, [detection(100, 100)], 0)
    tracks, next_id = analyzer._update_tracker(tracks, [detection(130, 110)], next_id)
    assert next_id == 1
    assert tracks[0]["frames_seen"] == 2 and tracks[0]["centroid"] == (130, 110)


def test_a_distant_detection_starts_a_new_track(analyzer):
    tracks, next_id = analyzer._update_tracker({}, [detection(100, 100)], 0)
    tracks, next_id = analyzer._update_tracker(tracks, [detection(300, 100)], next_id)
    assert next_id == 2
    assert tracks[0]["frames_since_seen"] == 1 and not tracks[0]["just_seen"]
    assert tracks[1]["just_seen"]


def test_each_detection_continues_at_most_one_track(analyzer):
    tracks, next_id = analyzer._update_tracker({}, [detection(100, 100), detection(140, 100)], 0)
    tracks, next_id = analyzer._update_tracker(tracks, [detection(120, 100)], next_id)
    assert sorted(t["frames_seen"] for t in tracks.values()) == [1, 2]


def test_tracks_unseen_for_too_long_are_dropped(analyzer):
    tracks, next_id = analyzer._update_tracker({}, [detection(100, 100)], 0)
    for _ in range(analyzer.config["tracking_inactivity_frames"]):
        tracks, next_id = analyzer._update_tracker(tracks, [], next_id)
    assert 0 in tracks
    tracks, next_id = analyzer._update_tracker(tracks, [], next_id)
    assert tracks == {}
