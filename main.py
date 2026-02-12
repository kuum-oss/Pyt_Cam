# main.py
from config import AppConfig
from services.camera import WebcamSource
from services.detector import MediaPipeFaceDetector
from services.pose import GeometryPoseEstimator
from services.meme import FileSystemMemeProvider
from services.posture import PostureMonitor
from app import AttentionGuardian

if __name__ == "__main__":
    # 1. Настройки
    config = AppConfig(
        PITCH_THRESHOLD=0.35,  # Порог для "Метода глаз"
        MEME_FOLDER="memes",   # Папка с видео
        YOUTUBE_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    try:
        # 2. Инициализация сервисов
        camera = WebcamSource(config.CAMERA_INDEX)
        detector = MediaPipeFaceDetector()
        pose_estimator = GeometryPoseEstimator()
        posture_monitor = PostureMonitor()
        meme_loader = FileSystemMemeProvider(config.MEME_FOLDER, config.YOUTUBE_URL)

        # 3. Сборка приложения
        # ВАЖНО: Порядок должен строго совпадать с __init__ в app.py:
        # (config, source, detector, estimator, posture_monitor, meme_provider)
        app = AttentionGuardian(
            config,
            camera,
            detector,
            pose_estimator,
            posture_monitor,
            meme_loader
        )

        # 4. Поехали!
        app.run()

    except Exception as e:
        print(f"Critical Error: {e}")