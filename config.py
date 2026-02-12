import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # НОВЫЙ ПОРОГ: Отношение (Нос-Рот) / (Глаз-Глаз)
    # Если значение меньше 0.35, значит лицо "сплющено" по вертикали (взгляд вниз)
    PITCH_THRESHOLD: float = 0.35

    # Путь к папке (попробуем на Рабочий стол, если meme.py позволит)
    MEME_FOLDER: str = "~/Desktop/Senior_Memes"

    YOUTUBE_URL: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    CAMERA_INDEX: int = 0
    WINDOW_NAME: str = "Senior Eye Tracker"