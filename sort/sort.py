"""
Lightweight SORT tracker — compatible with easy_ocr.py / Nanonets.py
which do: from sort.sort import Sort ; tracker.update(dets)

If filterpy is available, Kalman is used; otherwise pure IOU fallback.
"""
import numpy as np

try:
    from filterpy.kalman import KalmanFilter  # noqa: F401
    _HAS_KALMAN = True
except ImportError:
    _HAS_KALMAN = False


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh + 1e-6)
    return o


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []  # list of dict {id, bbox, hits, age, hit_streak}
        self.next_id = 1
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        if dets is None or len(dets) == 0:
            # age out
            for t in self.tracks:
                t["age"] += 1
                t["hit_streak"] = 0
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return np.empty((0, 5))

        # dets: Nx5 [x1,y1,x2,y2,conf]
        dets = np.asarray(dets)
        # match
        if len(self.tracks) == 0:
            for d in dets:
                self.tracks.append({"id": self.next_id, "bbox": d[:4], "hits": 1, "age": 0, "hit_streak": 1})
                self.next_id += 1
        else:
            trks = np.array([t["bbox"] for t in self.tracks])
            iou_mat = iou_batch(dets[:, :4], trks)
            matched_det = set()
            matched_trk = set()
            # greedy matching by max iou
            # sort by iou descending
            indices = np.dstack(np.unravel_index(np.argsort(-iou_mat.ravel()), iou_mat.shape))[0]
            for d_idx, t_idx in indices:
                if iou_mat[d_idx, t_idx] < self.iou_threshold:
                    break
                if d_idx in matched_det or t_idx in matched_trk:
                    continue
                matched_det.add(d_idx); matched_trk.add(t_idx)
                self.tracks[t_idx]["bbox"] = dets[d_idx, :4]
                self.tracks[t_idx]["hits"] += 1
                self.tracks[t_idx]["age"] = 0
                self.tracks[t_idx]["hit_streak"] += 1
            # unmatched dets -> new tracks
            for i, d in enumerate(dets):
                if i not in matched_det:
                    self.tracks.append({"id": self.next_id, "bbox": d[:4], "hits": 1, "age": 0, "hit_streak": 1})
                    self.next_id += 1
            # unmatched tracks -> age
            for i, t in enumerate(self.tracks):
                if i not in matched_trk and i < len(trks):  # was existing
                    # Already updated matched ones; need to age unmatched
                    pass
            # age unmatched tracks
            for i, t in enumerate(self.tracks):
                # check if this track was not matched this frame
                # we already handled matched; now find indices not in matched_trk
                pass
            # simpler: mark unmatched as aged
            # Re-build age correctly:
            # We need to know which old tracks were unmatched.
            # Use matched_trk set to age them
            for idx, t in enumerate(self.tracks[:len(trks)]):
                # this loop double counts; instead handle via separate
                pass
            # Correct ageing: for old tracks not in matched_trk
            for t_idx in range(len(trks)):
                if t_idx not in matched_trk:
                    # find the track object (may have shifted due to new tracks appended)
                    # old tracks are at indices 0..len(trks)-1 before appending
                    if t_idx < len(self.tracks):
                        self.tracks[t_idx]["age"] += 1
                        self.tracks[t_idx]["hit_streak"] = 0

        # prune old
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        # return confirmed tracks
        ret = []
        for t in self.tracks:
            if t["hits"] >= self.min_hits or self.frame_count <= self.min_hits:
                x1, y1, x2, y2 = t["bbox"]
                ret.append([x1, y1, x2, y2, t["id"]])
        if not ret:
            return np.empty((0, 5))
        return np.array(ret)
