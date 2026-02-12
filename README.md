# 🛡️ Senior Eye Tracker & Posture Guardian

![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge\&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black?style=for-the-badge\&logo=apple)
![M1 Optimized](https://img.shields.io/badge/M1%20Optimized-Yes-success?style=for-the-badge)

---

**Senior Eye Tracker** — AI-ассистент для людей, которые проводят за компьютером слишком много времени и расплачиваются за это шеей и спиной. Приложение в реальном времени отслеживает положение головы и осанку. При отклонении от нормы система немедленно реагирует и напоминает пользователю выпрямиться через визуальное уведомление, мем или видео.

---

## ✨ Основные функции

![Realtime](https://img.shields.io/badge/Realtime-Tracking-blue?style=for-the-badge)
![Posture](https://img.shields.io/badge/Posture-Detection-critical?style=for-the-badge)
![Calibration](https://img.shields.io/badge/Auto-Calibration-5s-success?style=for-the-badge)
![HUD](https://img.shields.io/badge/HUD-Live%20Metrics-purple?style=for-the-badge)

* **Real-time Monitoring**
  Отслеживание положения глаз и виртуального скелета (шея, плечи).

* **Auto-Calibration**
  Автоматическая калибровка рабочего положения за 5 секунд при каждом запуске.

* **HUD Interface**
  Отображение метрик `BODY` (осанка) и `TILT` (наклон головы) прямо в окне камеры.

* **Smart Alerts**
  Если некорректная поза удерживается более 2.5 секунд, срабатывает экстренная реакция.

* **Apple Silicon Optimized**
  Используется стабильный `MediaPipe Face Detection` без конфликтов на M1/M2.

---

## 🛠️ Как это работает

![Virtual Skeleton](https://img.shields.io/badge/Virtual-Skeleton-informational?style=for-the-badge)
![Math Model](https://img.shields.io/badge/Math-Based-Model-blue?style=for-the-badge)
![No Pose](https://img.shields.io/badge/MediaPipe%20Pose-NOT%20USED-red?style=for-the-badge)

Проект использует математическую модель **Виртуального Скелета**. Поскольку стандартный `MediaPipe Pose` нестабилен на macOS, положение плеч и наклон головы вычисляются относительно лица и глаз.

### Цветовая индикация

![Good](https://img.shields.io/badge/Posture-Good-brightgreen?style=for-the-badge)
![Bad](https://img.shields.io/badge/Posture-Bad-red?style=for-the-badge)

* **Зелёный** — осанка в норме (`Score > 0.88`)
* **Красный** — обнаружена сутулость или критический наклон головы

---

## 🚀 Установка и запуск

### 1. Требования

* Python 3.9+
* macOS (рекомендуется), Windows или Linux
* Веб-камера

---

### 2. Клонирование репозитория

```bash
git clone https://github.com/your-username/SeniorEyeTracker.git
cd SeniorEyeTracker
```

---

### 3. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Для Mac M1/M2 убедись, что корректно установлены `mediapipe` и `opencv-python`.

---

### 4. Настройка

* Помести свои видео (`.mp4`) в папку `memes/`
* Если папка пуста, будет использована резервная ссылка из `config.py`

---

### 5. Запуск

```bash
python main.py
```

---

## ⚙️ Настройка чувствительности

![Config](https://img.shields.io/badge/Config-config.py-lightgrey?style=for-the-badge)
![Threshold](https://img.shields.io/badge/Sensitivity-Adjustable-blue?style=for-the-badge)
![Memes](https://img.shields.io/badge/Memes-Custom%20MP4-purple?style=for-the-badge)

Все параметры находятся в `config.py`:

* `PITCH_THRESHOLD` — допустимый наклон головы (меньше = строже)
* `MEME_FOLDER` — путь к пользовательским видео
* `WINDOW_NAME` — имя окна приложения

---

## 📦 Структура проекта

```text
.
├── main.py        # Точка входа
├── app.py         # Управление состоянием и калибровкой
├── config.py      # Пороговые значения и пути
├── services/
│   ├── detector.py  # MediaPipe Face Detector
│   ├── posture.py   # Математика виртуального скелета
│   ├── camera.py    # Захват видео (macOS AVFoundation)
│   └── meme.py      # Логика "наказаний"
└── memes/         # Пользовательские видео
```

---

## 🤝 Контакты

Если проект реально помог твоей спине — поставь ⭐ репозиторию.
Если нет — значит, осанка всё ещё не приоритет.

Разработано с заботой о позвоночниках и инженерной строгостью.
