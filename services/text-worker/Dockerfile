FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY services/text-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/common ./common
COPY services/text-worker/*.py ./text-worker/
# Run as a normal user, not root; the model is downloaded into /models at startup.
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
WORKDIR /app/text-worker
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
