FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:0
ENV WAYLAND_DISPLAY=$WAYLAND_DISPLAY

# Установка системных зависимостей для Qt и OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    libfontconfig1 \
    libdbus-1-3 \
    libegl1 \
    xvfb \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*


# Предварительное создание каталога для X11-сокетов
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

# Настройка пользователя
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

# Запуск приложения с виртуальным дисплеем
CMD ["sh", "-c", "Xvfb :0 -screen 0 1024x768x16 & export DISPLAY=:0 && python neon_cicada/main.py"]
