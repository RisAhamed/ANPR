from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def track_plates(self, detections, frame):
        # detections: list of [x1, y1, x2, y2, conf]
        # frame: np.ndarray (not used for basic tracking)
        dets = []
        for d in detections:
            x1, y1, x2, y2, conf = d
            dets.append(([x1, y1, x2-x1, y2-y1], conf, None))  # xywh, conf, class
        tracks = self.tracker.update_tracks(dets, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results.append([x1, y1, x2, y2, track.track_id])
        return results
