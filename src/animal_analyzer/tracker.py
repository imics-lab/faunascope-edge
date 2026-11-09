import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def box_iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    label: str
    box: Box
    confidence: float
    hits: int = 1         # detections matched to this track
    missed: int = 0       # consecutive detection frames without a match
    detected: bool = True  # the box comes from the detector in the current frame
    points: Optional[np.ndarray] = None


class BoxTracker:
    """
    Follows detector boxes through the frames between detector runs, after FaunaCodec's ROI tracker (faunacodec/detection/detector.py).

    On a detection frame, each detection continues the track of the same label whose box overlaps it most (IoU of at least match_iou), and the rest start new tracks; a track unmatched in more than max_missed detection frames in a row ends. Between detection frames, each visible track's box moves with the median optical flow (Lucas-Kanade) of corner points inside it, and stays in place when too few points can be followed. A track is visible while its last detection frame matched it.
    """
    def __init__(self, match_iou: float, max_missed: int, max_points: int = 80, min_points: int = 10):
        self.match_iou = match_iou
        self.max_missed = max_missed
        self.max_points = max_points
        self.min_points = min_points
        self.tracks: List[Track] = []
        self.next_id = 1
        self.prev_gray = None

    def update(self, gray: np.ndarray, detections: Optional[List[Dict]]) -> List[Track]:
        """Advance to the next frame. detections is None between detection frames. Returns the visible tracks."""
        if detections is None:
            for t in self.visible():
                t.detected = False
                self._follow(t, gray)
        else:
            self._match(gray, detections)
        self.prev_gray = gray
        return self.visible()

    def visible(self) -> List[Track]:
        return [t for t in self.tracks if t.missed == 0]

    def _match(self, gray: np.ndarray, detections: List[Dict]) -> None:
        pairs = sorted(((box_iou(d['box_xyxy'], t.box), di, ti)
                        for di, d in enumerate(detections) for ti, t in enumerate(self.tracks)
                        if d['label'] == t.label), reverse=True)
        used_d, used_t = set(), set()
        for iou, di, ti in pairs:
            if iou < self.match_iou:
                break
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            t = self.tracks[ti]
            t.box, t.confidence = detections[di]['box_xyxy'], detections[di]['confidence']
            t.hits += 1
            t.missed = 0

        for ti, t in enumerate(self.tracks):
            t.detected = ti in used_t
            if not t.detected:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, d['label'], d['box_xyxy'], d['confidence']))
                self.next_id += 1

        for t in self.visible():
            t.points = self._corners(gray, t.box)

    def _corners(self, gray: np.ndarray, box: Box) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        mask = np.zeros_like(gray)
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=self.max_points, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=7)

    def _follow(self, t: Track, gray: np.ndarray) -> None:
        if t.points is None or len(t.points) < self.min_points:
            return
        points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, t.points, None, winSize=(21, 21), maxLevel=3)
        good = status.reshape(-1) == 1
        if good.sum() < self.min_points:
            t.points = None
            return
        dx, dy = np.median(points.reshape(-1, 2)[good] - t.points.reshape(-1, 2)[good], axis=0)
        h, w = gray.shape
        x1, y1, x2, y2 = t.box
        dx = float(np.clip(dx, -x1, w - x2))
        dy = float(np.clip(dy, -y1, h - y2))
        t.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        t.points = points[good].reshape(-1, 1, 2)
