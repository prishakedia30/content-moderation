FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/video-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the Whisper "base" weights into the image so pods start without downloading them.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1
COPY services/common ./common
COPY services/video-worker/*.py services/video-worker/wordlist.txt ./video-worker/
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app
WORKDIR /app/video-worker
EXPOSE 8000
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
