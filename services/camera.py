import cv2
import numpy as np
import sys
from typing import Optional
from interfaces import IFrameSource


class WebcamSource(IFrameSource):
    def __init__(self, camera_index: int):
        # macOS Specific: Используем CAP_AVFOUNDATION для нативности и скорости
        backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY

        self._cap = cv2.VideoCapture(camera_index, backend)

        # Оптимизация разрешения
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self._cap.isOpened():
            raise PermissionError(
                f"Could not open camera {camera_index}. "
                "Check macOS System Settings -> Privacy & Security -> Camera."
            )

    def get_frame(self) -> Optional[np.ndarray]:
        success, frame = self._cap.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()
        