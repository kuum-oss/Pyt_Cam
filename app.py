import cv2
import time
import numpy as np
from config import AppConfig
from interfaces import IFrameSource, IFaceDetector, IPoseEstimator, IContentProvider


class AttentionGuardian:
    def __init__(self, config: AppConfig, source: IFrameSource, detector: IFaceDetector,
                 estimator: IPoseEstimator, meme_provider: IContentProvider):
        self._config = config
        self._source = source
        self._detector = detector
        self._estimator = estimator  # Теперь это наш улучшенный класс
        self._meme_provider = meme_provider
        self._is_running = True

        # Состояние приложения
        self._state = "CALIBRATION"  # CALIBRATION -> MONITORING -> WARNING -> ALARM
        self._calibration_start = time.time()
        self._calibration_duration = 4.0  # Секунд на калибровку

        self._look_down_start_time = None
        self._last_action_time = 0

    def _draw_hud(self, frame, ratio, deviation, is_calibrating=False):
        h, w, _ = frame.shape

        if is_calibrating:
            # Отрисовка процесса калибровки
            elapsed = time.time() - self._calibration_start
            progress = min(1.0, elapsed / self._calibration_duration)

            bar_w = int(w * 0.6)
            cv2.rectangle(frame, (w // 2 - bar_w // 2, h // 2), (w // 2 + bar_w // 2, h // 2 + 30), (50, 50, 50), -1)
            cv2.rectangle(frame, (w // 2 - bar_w // 2, h // 2),
                          (w // 2 - bar_w // 2 + int(bar_w * progress), h // 2 + 30), (0, 255, 255), -1)
            cv2.putText(frame, "SIT STRAIGHT! CALIBRATING...", (w // 2 - 150, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return

        # Отрисовка в режиме мониторинга
        # Зеленая/Красная полоска уровня внимания
        bar_height = int(h * 0.3)
        # deviation 1.0 = отлично, 0.7 = плохо. Масштабируем для визуала.
        fill_level = int(max(0, min(1, (deviation - 0.5) * 2)) * bar_height)

        color = (0, 255, 0) if deviation > 0.8 else (0, 0, 255)

        # Фон бара
        cv2.rectangle(frame, (20, h // 2 - bar_height // 2), (50, h // 2 + bar_height // 2), (30, 30, 30), -1)
        # Уровень
        cv2.rectangle(frame, (20, h // 2 + bar_height // 2 - fill_level), (50, h // 2 + bar_height // 2), color, -1)
        cv2.rectangle(frame, (20, h // 2 - bar_height // 2), (50, h // 2 + bar_height // 2), (255, 255, 255), 2)

        cv2.putText(frame, f"{deviation:.0%}", (15, h // 2 + bar_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)

    def run(self):
        cv2.namedWindow(self._config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(self._config.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass

        print("[SYSTEM] Starting calibration sequence...")

        while self._is_running:
            frame = self._source.get_frame()
            if frame is None: break
            h, w, _ = frame.shape

            detection_result = self._detector.process(frame)

            # --- ЛОГИКА КАЛИБРОВКИ ---
            if self._state == "CALIBRATION":
                if detection_result and detection_result.detections:
                    for detection in detection_result.detections:
                        kp = detection.location_data.relative_keypoints
                        current_ratio = self._estimator.get_pitch_ratio(kp, w, h)
                        self._estimator.calibrate(current_ratio)  # Учим "эталон"

                self._draw_hud(frame, 0, 0, is_calibrating=True)

                if time.time() - self._calibration_start > self._calibration_duration:
                    print("[SYSTEM] Calibration DONE. Monitoring active.")
                    self._state = "MONITORING"

                cv2.imshow(self._config.WINDOW_NAME, frame)
                if cv2.waitKey(5) & 0xFF == ord('q'): break
                continue

            # --- ЛОГИКА МОНИТОРИНГА ---
            is_looking_down = False
            deviation = 1.0

            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    kp = detection.location_data.relative_keypoints

                    # Получаем уже сглаженное значение
                    current_ratio = self._estimator.get_pitch_ratio(kp, w, h)

                    # Насколько мы отклонились от эталона (1.0 = идеал, < 0.8 = наклон)
                    deviation = self._estimator.get_deviation()

                    # Рисуем "Senior" UI - вектора
                    nose_x, nose_y = int(kp[2].x * w), int(kp[2].y * h)
                    mouth_x, mouth_y = int(kp[3].x * w), int(kp[3].y * h)
                    cv2.line(frame, (nose_x, nose_y), (mouth_x, mouth_y), (255, 255, 0), 2)

                    # Триггер: если текущее соотношение упало ниже 80% от эталона
                    if deviation < 0.82:  # Чувствительность (можно вынести в конфиг)
                        is_looking_down = True
            else:
                is_looking_down = True  # Нет лица = нет работы

            # Обработка таймеров и наказаний
            if is_looking_down:
                if self._look_down_start_time is None:
                    self._look_down_start_time = time.time()

                elapsed = time.time() - self._look_down_start_time
                remaining = 2.0 - elapsed

                if remaining > 0:
                    # Предупреждение
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2 * (1 - remaining / 2), frame, 1.0, 0, frame)
                    cv2.putText(frame, "EYES UP!", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                                3)
                else:
                    # НАКАЗАНИЕ
                    cv2.putText(frame, "PUNISHMENT!", (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                5)
                    if time.time() - self._last_action_time > 15:
                        self._meme_provider.play_emergency_action()
                        self._last_action_time = time.time()
            else:
                self._look_down_start_time = None

            self._draw_hud(frame, 0, deviation, is_calibrating=False)

            cv2.imshow(self._config.WINDOW_NAME, frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                self._is_running = False

        self._source.release()
        cv2.destroyAllWindows()