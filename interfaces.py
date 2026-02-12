from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class IFrameSource(ABC):
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

class IFaceDetector(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> object:
        pass

class IPoseEstimator(ABC):
    @abstractmethod
    def get_pitch(self, face_landmarks, image_shape) -> float:
        pass

class IContentProvider(ABC):
    @abstractmethod
    def get_content_frame(self) -> Optional[np.ndarray]:
        pass