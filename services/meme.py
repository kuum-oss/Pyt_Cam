import cv2
import numpy as np
import os
import random
import webbrowser
from pathlib import Path
from typing import Optional
from interfaces import IContentProvider


class FileSystemMemeProvider(IContentProvider):
    def __init__(self, folder: str, youtube_url: str):
        # Пытаемся развернуть путь (превратить ~/Desktop в полный путь)
        self._folder = Path(folder).expanduser().resolve()
        self._youtube_url = youtube_url

        # Логика создания папки с защитой от ошибок
        try:
            if not self._folder.exists():
                self._folder.mkdir(parents=True, exist_ok=True)
                print(f"\n[SUCCESS] Папка создана! Закиньте видео сюда:\n---> {self._folder}\n")
            else:
                print(f"\n[INFO] Папка с мемами найдена:\n---> {self._folder}\n")
        except PermissionError:
            # Если macOS не дает писать на рабочий стол, создаем папку memes внутри проекта
            print(f"[ERROR] Нет прав на создание папки на Рабочем столе. Создаю в папке проекта.")
            self._folder = Path("memes").resolve()
            self._folder.mkdir(exist_ok=True)
            print(f"---> Новая папка: {self._folder}\n")

    def get_random_video(self) -> Optional[Path]:
        # Ищем mp4, mov, avi
        extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = [f for f in self._folder.iterdir() if f.suffix.lower() in extensions]
        return random.choice(videos) if videos else None

    def play_emergency_action(self):
        video_path = self.get_random_video()
        if video_path:
            print(f"[ACTION] Запускаю видео: {video_path.name}")
            # Универсальная команда открытия для macOS
            os.system(f"open '{video_path}'")
        else:
            print("[ACTION] Видео не найдены. Открываю YouTube.")
            webbrowser.open(self._youtube_url)

    def get_content_frame(self) -> Optional[np.ndarray]:
        return None