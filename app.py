import cv2
from config import AppConfig
from interfaces import IFrameSource, IFaceDetector, IPoseEstimator, IContentProvider


class AttentionGuardian:
    def __init__(
            self,
            config: AppConfig,
            source: IFrameSource,
            detector: IFaceDetector,
            estimator: IPoseEstimator,
            meme_provider: IContentProvider
    ):
        self._config = config
        self._source = source
        self._detector = detector
        self._estimator = estimator
        self._meme_provider = meme_provider
        self._is_running = True

    def run(self):
        print(f"Starting {self._config.WINDOW_NAME}...")
        print("Press 'q' to exit.")

        while self._is_running:
            frame = self._source.get_frame()
            if frame is None:
                break

            img_h, img_w, _ = frame.shape

            # 1. Детекция (используем Face Detection)
            detection_result = self._detector.process(frame)

            show_meme = False
            pitch_indicator = 0.0

            # Новая логика обработки для Face Detection
            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    # Извлекаем относительные ключевые точки
                    # Индексы MediaPipe Face Detection: 2 - кончик носа, 3 - центр рта
                    keypoints = detection.location_data.relative_keypoints

                    nose_y = keypoints[2].y
                    mouth_y = keypoints[3].y

                    # Вычисляем индикатор наклона (расстояние между носом и ртом)
                    # Когда голова наклонена вниз, это расстояние сокращается
                    pitch_indicator = mouth_y - nose_y

                    # Порог срабатывания: если расстояние меньше 0.06 (подберите экспериментально)
                    # В FaceMesh мы использовали градусы, здесь — относительные координаты
                    if pitch_indicator < 0.065:
                        show_meme = True

            # 2. Рендер результата
            final_frame = frame

            if show_meme:
                meme_frame = self._meme_provider.get_content_frame()
                if meme_frame is not None:
                    final_frame = cv2.resize(meme_frame, (img_w, img_h))
                    cv2.putText(final_frame, "GET BACK TO WORK!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Отрисовка статуса (только если не показываем мем)
            if not show_meme:
                color = (0, 255, 0)
                status_text = f"Face Gap: {pitch_indicator:.3f}"
                cv2.putText(final_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(self._config.WINDOW_NAME, final_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                self._is_running = False

        self._source.release()
        cv2.destroyAllWindows()