FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser
COPY src/ ./src/
COPY docs/ ./docs/
COPY reports/metrics.json reports/data_quality.json reports/data_manifest.json ./reports/
COPY README.md ./
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "src/api.py", "--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/models"]
