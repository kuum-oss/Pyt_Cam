from config import AppConfig
from services.camera import WebcamSource
from services.detector import MediaPipeFaceDetector
from services.pose import GeometryPoseEstimator
from services.meme import FileSystemMemeProvider
from app import AttentionGuardian

if __name__ == "__main__":
    # Теперь передаем только те параметры, которые есть в config.py
    config = AppConfig(
        PITCH_THRESHOLD=0.065,  # Порог для Face Detection
        MEME_FOLDER="memes",
        YOUTUBE_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    try:
        camera = WebcamSource(config.CAMERA_INDEX)
        detector = MediaPipeFaceDetector()
        pose_estimator = GeometryPoseEstimator()

        # Передаем папку и ссылку в провайдер контента
        meme_loader = FileSystemMemeProvider(config.MEME_FOLDER, config.YOUTUBE_URL)

        app = AttentionGuardian(config, camera, detector, pose_estimator, meme_loader)
        app.run()

    except Exception as e:
        print(f"Critical Error: {e}")