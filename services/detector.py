import cv2
import mediapipe as mp
import numpy as np
from interfaces import IFaceDetector

class MediaPipeFaceDetector(IFaceDetector):
    def __init__(self):
        # Используем Face Detection вместо Face Mesh для стабильности на Mac
        self._mp_face_detection = mp.solutions.face_detection
        self._face_detection = self._mp_face_detection.FaceDetection(
            model_selection=0, # 0 для селфи-расстояния (до 2 метров)
            min_detection_confidence=0.5
        )

    def process(self, image: np.ndarray):
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Возвращаем результат детекции
        return self._face_detection.process(image_rgb)