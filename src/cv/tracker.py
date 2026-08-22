"""SORT (Simple Online and Realtime Tracking) with a Kalman filter.

Assigns stable identities to detections across frames so the studio can
navigate time coherently and attach conditioning motion to real objects.
"""

from __future__ import annotations

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    from scipy.optimize import linear_sum_assignment
    _HAS = True
except Exception:  # pragma: no cover - optional deps
    _HAS = False


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    wh = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    b = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    return wh / (a + b - wh + 1e-9)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        return np.array([x, y, w * h, w / h]).reshape((4, 1))

    def to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        box = [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2]
        return np.array(box if score is None else box + [score]).reshape((1, -1))

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.to_bbox(self.kf.x))
        return self.history[-1]

    def state(self):
        return self.to_bbox(self.kf.x)


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        if not _HAS:
            return np.empty((0, 5))
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, u_dets, u_trks = self._associate(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        for i in u_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.state()[0]
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.concatenate(ret) if ret else np.empty((0, 5))

    def _associate(self, detections, trackers, iou_threshold=None):
        iou_threshold = iou_threshold or self.iou_threshold
        if len(trackers) == 0:
            return np.empty((0, 2), int), np.arange(len(detections)), np.empty((0, 5), int)
        iou_matrix = iou_batch(detections.astype(float), trackers.astype(float))
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(int)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched = np.stack(np.where(a), axis=1)
            else:
                row, col = linear_sum_assignment(-iou_matrix)
                matched = np.asarray(list(zip(row, col)))
        else:
            matched = np.empty((0, 2))

        unmatched_d, unmatched_t = [], []
        for d in range(len(detections)):
            if d not in matched[:, 0]:
                unmatched_d.append(d)
        for t in range(len(trackers)):
            if t not in matched[:, 1]:
                unmatched_t.append(t)

        matches = []
        for m in matched:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_d.append(m[0])
                unmatched_t.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        return (np.concatenate(matches, 0) if matches else np.empty((0, 2), int),
                np.array(unmatched_d, int), np.array(unmatched_t, int))