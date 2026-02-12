import numpy as np
import cv2


class PostureMonitor:
    def __init__(self, smoothing=0.3):
        self._alpha = smoothing
        self._base_y = None  # Эталон высоты носа
        self._base_dist = None  # Эталон масштаба (между глазами)
        self._curr_y = 0.0
        self._curr_dist = 0.0

    def calibrate(self, keypoints, h, w):
        # 0: левый глаз, 1: правый глаз, 2: нос
        eye_dist = np.linalg.norm(np.array([keypoints[0].x - keypoints[1].x])) * w
        nose_y = keypoints[2].y * h

        if self._base_y is None:
            self._base_y, self._base_dist = nose_y, eye_dist
        else:
            self._base_y = self._base_y * 0.8 + nose_y * 0.2
            self._base_dist = self._base_dist * 0.8 + eye_dist * 0.2

    def update(self, keypoints, h, w):
        eye_dist = np.linalg.norm(np.array([keypoints[0].x - keypoints[1].x])) * w
        nose_y = keypoints[2].y * h
        # Сглаживание EMA
        self._curr_y = (self._alpha * nose_y) + ((1 - self._alpha) * self._curr_y)
        self._curr_dist = (self._alpha * eye_dist) + ((1 - self._alpha) * self._curr_dist)

    def get_metrics(self):
        if self._base_y is None or self._base_dist == 0: return 1.0
        # Считаем просадку: насколько нос упал вниз относительно масштаба лица
        drop = (self._curr_y - self._base_y) / (self._base_dist + 0.1)
        return 1.0 - drop

    def draw_skeleton(self, frame, keypoints):
        h, w, _ = frame.shape

        def pt(idx): return int(keypoints[idx].x * w), int(keypoints[idx].y * h)

        nose, mouth = pt(2), pt(3)
        l_eye, r_eye = pt(0), pt(1)

        # Виртуальные плечи (математически ниже подбородка)
        sh_y = mouth[1] + int(h * 0.12)
        l_sh, r_sh = (nose[0] - 80, sh_y), (nose[0] + 80, sh_y)

        score = self.get_metrics()
        color = (0, 255, 0) if score > 0.85 else (0, 0, 255)

        # Рисуем "Палки"
        cv2.line(frame, l_eye, r_eye, color, 2)  # Глаза
        cv2.line(frame, nose, mouth, color, 3)  # Шея
        cv2.line(frame, l_sh, r_sh, color, 5)  # Плечи
        cv2.circle(frame, nose, 5, (0, 255, 255), -1)
        return score, color