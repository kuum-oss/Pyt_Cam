import cv2
import time
from config import AppConfig
from interfaces import IFrameSource, IFaceDetector, IPoseEstimator, IContentProvider


class AttentionGuardian:
    def __init__(self, config: AppConfig, source: IFrameSource, detector: IFaceDetector,
                 estimator: IPoseEstimator, posture_monitor, meme_provider: IContentProvider):
        self._config = config
        self._source = source
        self._detector = detector
        self._estimator = estimator  # Теперь используется для Pitch Ratio
        self._posture = posture_monitor
        self._meme_provider = meme_provider
        self._is_running = True
        self._state = "CALIBRATION"
        self._cal_start = time.time()
        self._bad_time = None
        self._last_action = 0
        self._base_pitch = 1.0

    def run(self):
        cv2.namedWindow(self._config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        while self._is_running:
            frame = self._source.get_frame()
            if frame is None: break
            h, w, _ = frame.shape

            # Инициализируем переменную, чтобы избежать ошибки "referenced before assignment"
            res = self._detector.process(frame)
            fail_msg = None
            body_score = 1.0
            pitch_ratio = 1.0

            if res and res.detections:
                kp = res.detections[0].location_data.relative_keypoints
                current_pitch = self._estimator.get_pitch(kp, frame.shape)  # Вычисляем текущий наклон

                if self._state == "CALIBRATION":
                    self._posture.calibrate(kp, h, w)
                    self._estimator.calibrate(current_pitch)  # Сохраняем эталон наклона

                    elapsed = time.time() - self._cal_start
                    cv2.putText(frame, f"SIT STRAIGHT: {5 - int(elapsed)}s", (w // 2 - 180, h // 2),
                                1, 2, (0, 255, 255), 2)
                    if elapsed > 5:
                        self._state = "MONITOR"
                else:
                    self._posture.update(kp, h, w)
                    body_score, color = self._posture.draw_skeleton(frame, kp)

                    # Получаем отклонение в % от калибровки
                    pitch_ratio = self._estimator.get_deviation_ratio(current_pitch)

                    if pitch_ratio < self._config.PITCH_THRESHOLD:
                        fail_msg = "LIFT YOUR EYES!"
                    elif body_score < 0.88:
                        fail_msg = "SIT STRAIGHT!"
            else:
                fail_msg = "SEARCHING..."

            # --- HUD (Отображение метрик) ---
            cv2.rectangle(frame, (5, 5), (200, 85), (0, 0, 0), -1)
            cv2.putText(frame, f"BODY: {body_score:.2f}", (10, 35), 1, 1.3, (0, 255, 0), 2)
            cv2.putText(frame, f"TILT: {pitch_ratio:.2f}", (10, 65), 1, 1, (200, 200, 200), 1)

            if fail_msg and self._state == "MONITOR":
                if self._bad_time is None: self._bad_time = time.time()
                cv2.putText(frame, fail_msg, (w // 2 - 150, h // 2), 1, 3, (0, 0, 255), 4)
                if time.time() - self._bad_time > 2.5:
                    if time.time() - self._last_action > 15:
                        self._meme_provider.play_emergency_action()
                        self._last_action = time.time()
            else:
                self._bad_time = None

            cv2.imshow(self._config.WINDOW_NAME, frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): self._is_running = False

        self._source.release()
        cv2.destroyAllWindows()