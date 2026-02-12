from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # Порог отклонения: 0.85 означает, что если голова наклонилась
    # более чем на 15% от откалиброванной нормы, сработает алерт.
    PITCH_THRESHOLD: float = 0.85

    MEME_FOLDER: str = "~/Desktop/Senior_Memes"
    YOUTUBE_URL: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    CAMERA_INDEX: int = 0
    WINDOW_NAME: str = "Senior Eye Tracker"