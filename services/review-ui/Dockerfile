FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY services/review-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/review-ui/app.py .
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--server.baseUrlPath", "review", "--browser.gatherUsageStats", "false"]
