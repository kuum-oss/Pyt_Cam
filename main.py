from config import AppConfig
from services.camera import WebcamSource
from services.detector import MediaPipeFaceDetector
from services.pose import GeometryPoseEstimator
from services.meme import FileSystemMemeProvider
from app import AttentionGuardian

if __name__ == "__main__":
    # Dependency Injection Container (Manual)

    config = AppConfig(
        PITCH_THRESHOLD=15.0,
        MEME_FOLDER="memes",
        TARGET_MEME_NAME="clean_code"
    )

    try:
        # Инициализация с учетом импортов из пакета services
        camera = WebcamSource(config.CAMERA_INDEX)
        detector = MediaPipeFaceDetector()
        pose_estimator = GeometryPoseEstimator()
        meme_loader = FileSystemMemeProvider(config.MEME_FOLDER, config.TARGET_MEME_NAME)

        app = AttentionGuardian(config, camera, detector, pose_estimator, meme_loader)
        app.run()

    except Exception as e:
        print(f"Critical Error: {e}")