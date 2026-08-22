"""
DeepSORT tracker wrapper — uses deep-sort-realtime if available,
falls back to SORT-like simple IOU tracker so the app never crashes.
"""
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
    _HAS_DEEPSORT = True
except ImportError:
    _HAS_DEEPSORT = False
    logger.warning("deep-sort-realtime not installed, using fallback IOU tracker")


class _FallbackTracker:
    """Simple IOU tracker — no appearance features, but deterministic."""
    def __init__(self, iou_threshold=0.3, max_age=20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}  # id -> {bbox, age}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0
        area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, dets: np.ndarray, frame=None):
        # dets: Nx5 [x1,y1,x2,y2,conf]
        if dets is None or len(dets) == 0:
            # age out
            for tid in list(self.tracks):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
            return np.empty((0, 5))
        results = []
        used = set()
        for det in dets:
            x1,y1,x2,y2,conf = det[:5]
            best_iou, best_id = 0, None
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                iou = self._iou((x1,y1,x2,y2), t["bbox"][:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                self.tracks[best_id] = {"bbox": (x1,y1,x2,y2,conf), "age": 0}
                results.append([x1,y1,x2,y2,best_id])
                used.add(best_id)
            else:
                nid = self.next_id; self.next_id += 1
                self.tracks[nid] = {"bbox": (x1,y1,x2,y2,conf), "age": 0}
                results.append([x1,y1,x2,y2,nid])
        # age unmatched
        for tid in list(self.tracks):
            if tid not in [r[4] for r in results]:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return np.array(results) if results else np.empty((0,5))


class DeepSortTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3):
        if _HAS_DEEPSORT:
            self.tracker = _DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=nms_max_overlap,
                max_cosine_distance=max_cosine_distance,
                embedder="mobilenet",
                half=True,
            )
            self._use_deep = True
        else:
            self.tracker = _FallbackTracker(iou_threshold=0.3, max_age=max_age)
            self._use_deep = False

    def track_plates(self, boxes: List[Tuple[int,int,int,int,float]], frame) -> List[Tuple[int,int,int,int,int]]:
        """Input: boxes as (x1,y1,x2,y2,conf). Output: (x1,y1,x2,y2,track_id)"""
        if not boxes:
            # still need to age tracks
            if not self._use_deep:
                self.tracker.update(np.empty((0,5)), frame)
            return []
        if self._use_deep:
            # deep_sort_realtime expects [ [x1,y1,x2,y2], conf, class_id ]
            dets = [ ([int(b[0]), int(b[1]), int(b[2]), int(b[3])], float(b[4]), "plate") for b in boxes ]
            tracks = self.tracker.update_tracks(dets, frame=frame)
            out = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = int(t.track_id)
                ltrb = t.to_ltrb()  # [x1,y1,x2,y2]
                x1,y1,x2,y2 = map(int, ltrb)
                out.append((x1,y1,x2,y2,tid))
            return out
        else:
            arr = np.array([[b[0],b[1],b[2],b[3],b[4]] for b in boxes], dtype=float)
            res = self.tracker.update(arr, frame)
            return [tuple(map(int, r)) for r in res]  # type: ignore

    # backwards compat alias
    def update(self, *a, **kw):
        return self.track_plates(*a, **kw)
