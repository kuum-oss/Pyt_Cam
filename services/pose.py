import numpy as np
from interfaces import IPoseEstimator

class GeometryPoseEstimator(IPoseEstimator):
    def __init__(self, smoothing_factor=0.2):
        self._alpha = smoothing_factor
        self._filtered_pitch = None
        self._baseline_pitch = None

    def get_pitch(self, keypoints, image_shape) -> float:
        h, w, _ = image_shape
        def to_px(kp): return np.array([kp.x * w, kp.y * h])

        # 0:L.Eye, 1:R.Eye, 2:Nose, 3:Mouth
        eye_dist = np.linalg.norm(to_px(keypoints[0]) - to_px(keypoints[1]))
        nose_mouth_dist = np.linalg.norm(to_px(keypoints[2]) - to_px(keypoints[3]))

        raw_pitch = nose_mouth_dist / (eye_dist + 0.1)

        if self._filtered_pitch is None:
            self._filtered_pitch = raw_pitch
        else:
            self._filtered_pitch = (self._alpha * raw_pitch) + ((1 - self._alpha) * self._filtered_pitch)
        return self._filtered_pitch

    def calibrate(self, current_pitch: float):
        self._baseline_pitch = current_pitch

    def get_deviation_ratio(self, current_pitch: float) -> float:
        if self._baseline_pitch is None or self._baseline_pitch == 0:
            return 1.0
        return current_pitch / self._baseline_pitch