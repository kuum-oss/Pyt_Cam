from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Чувствительность наклона головы (градусы).
    # Если угол больше этого (голова вниз), включается мем.
    PITCH_THRESHOLD: float = 15.0

    # Путь к папке с мемами
    MEME_FOLDER: str = "./memes"
    # Часть названия файла, который ищем
    TARGET_MEME_NAME: str = "clean_code"

    # Индекс камеры (0 или 1)
    CAMERA_INDEX: int = 0
    WINDOW_NAME: str = "Senior Eye Tracker (macOS)"