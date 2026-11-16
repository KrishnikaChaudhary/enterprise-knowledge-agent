# Small, non-root image. Cloud Run injects $PORT and a service-account identity (no keys in the image).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 KB_DB=/tmp/kb.db KB_CORPUS_DIR=/app/data/corpus
WORKDIR /app
COPY pyproject.toml README.md ./
COPY kagent ./kagent
COPY data/corpus ./data/corpus
RUN pip install --no-cache-dir . && useradd --system --uid 10001 app
USER app
EXPOSE 8080
CMD ["python", "-m", "kagent.server"]
