import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional
from interfaces import IContentProvider


class FileSystemMemeProvider(IContentProvider):
    def __init__(self, folder: str, target_name: str):
        self._folder = Path(folder)
        self._target_name = target_name
        self._meme_path = self._find_meme()
        self._meme_image = self._load_meme()

    def _find_meme(self) -> Optional[Path]:
        if not self._folder.exists():
            os.makedirs(self._folder, exist_ok=True)
            print(f"[Warning] Created meme folder at {self._folder}. Put an image there!")
            return None

        for file in self._folder.iterdir():
            if self._target_name in file.name and file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                return file
        return None

    def _load_meme(self) -> Optional[np.ndarray]:
        if self._meme_path:
            img = cv2.imread(str(self._meme_path))
            if img is None:
                print(f"[Error] Failed to load image: {self._meme_path}")
            return img
        return None

    def get_content_frame(self) -> Optional[np.ndarray]:
        if self._meme_image is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "MEME NOT FOUND", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(blank, f"Put '{self._target_name}' in folder", (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            return blank
        return self._meme_image
    