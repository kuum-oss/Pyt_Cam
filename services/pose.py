import cv2
import numpy as np
from interfaces import IPoseEstimator


class GeometryPoseEstimator(IPoseEstimator):
    """
    Вычисляет наклон головы используя PnP (Perspective-n-Point).
    """

    def get_pitch(self, face_landmarks, image_shape) -> float:
        img_h, img_w, _ = image_shape

        face_3d = []
        face_2d = []

        # Ключевые точки: Нос(1), Подбородок(152), Левый глаз(33), Правый глаз(263), Рот лев(61), Рот прав(291)
        key_landmarks = [1, 152, 33, 263, 61, 291]

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in key_landmarks:
                if idx == 1:  # Нос
                    face_3d.append([lm.x * img_w, lm.y * img_h, lm.z * 3000])
                else:
                    face_3d.append(
                        [lm.x * img_w, lm.y * img_h, lm.z])  # Для остальных z берем как есть для упрощения или 0

                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])

        # Исправляем размерность face_3d, если мы заполняли его не полностью в цикле выше
        # (в оригинале логика была чуть сложнее с if idx==1, здесь упростим для надежности)
        # Пересобираем face_3d правильно:
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in key_landmarks:
                if idx == 1:
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
                face_3d.append([int(lm.x * img_w), int(lm.y * img_h), lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if not success:
            return 0.0

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return angles[0] * 360