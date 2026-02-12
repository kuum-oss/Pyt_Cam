import numpy as np
from interfaces import IPoseEstimator

class GeometryPoseEstimator(IPoseEstimator):
    """
    Продвинутый эстиматор с подавлением шума и нормализацией.
    """
    def __init__(self, smoothing_factor=0.3):
        # Коэффициент сглаживания (0.1 - очень плавно, 0.9 - резко)
        self._alpha = smoothing_factor
        self._filtered_ratio = None
        self._baseline_ratio = None # Эталонное значение (когда человек сидит ровно)

    def calibrate(self, current_ratio: float):
        """Устанавливает базовую линию (калибровка)"""
        if self._baseline_ratio is None:
            self._baseline_ratio = current_ratio
        else:
            # Медленно подстраиваемся под среднее значение во время калибровки
            self._baseline_ratio = self._baseline_ratio * 0.95 + current_ratio * 0.05

    def get_pitch_ratio(self, keypoints, width, height) -> float:
        """
        Возвращает нормализованный коэффициент наклона.
        Чем МЕНЬШЕ число, тем сильнее наклон вниз.
        """
        # Преобразуем относительные координаты в пиксели
        def to_px(kp): return np.array([kp.x * width, kp.y * height])

        # Индексы FaceDetection: 0-L.Eye, 1-R.Eye, 2-Nose, 3-Mouth
        l_eye = to_px(keypoints[0])
        r_eye = to_px(keypoints[1])
        nose = to_px(keypoints[2])
        mouth = to_px(keypoints[3])

        # 1. Вектор глаз (горизонт)
        eye_vector = np.linalg.norm(r_eye - l_eye)
        if eye_vector < 1: eye_vector = 1 # Защита от деления на 0

        # 2. Вектор Нос-Рот (вертикаль)
        # Используем именно евклидово расстояние, чтобы наклон головы ВБОК не ломал логику
        nose_mouth_vector = np.linalg.norm(mouth - nose)

        # 3. Сырой коэффициент (Ratio)
        raw_ratio = nose_mouth_vector / eye_vector

        # 4. Фильтрация (EMA Filter)
        if self._filtered_ratio is None:
            self._filtered_ratio = raw_ratio
        else:
            self._filtered_ratio = (self._alpha * raw_ratio) + ((1 - self._alpha) * self._filtered_ratio)

        return self._filtered_ratio

    def get_deviation(self) -> float:
        """Показывает, насколько текущее положение отклонилось от эталона в %"""
        if self._baseline_ratio is None or self._filtered_ratio is None:
            return 0.0
        # Если 1.0 -> мы сидим ровно. Если 0.7 -> мы наклонились на 30%
        return self._filtered_ratio / self._baseline_ratio

    def get_pitch(self, face_landmarks, image_shape) -> float:
        # Заглушка для интерфейса, если он требует старый метод
        return 0.0