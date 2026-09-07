FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    POLICY_PULSE_DATA_DIR=/var/lib/policy-pulse/data \
    POLICY_PULSE_LOG_DIR=/var/lib/policy-pulse/logs
ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
ENV BUILD_COMMIT=${BUILD_COMMIT} BUILD_DATE=${BUILD_DATE}
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && useradd --uid 10001 --create-home appuser \
    && mkdir -p /var/lib/policy-pulse/data /var/lib/policy-pulse/logs \
    && chown -R appuser:appuser /var/lib/policy-pulse
COPY backend ./backend
COPY app/__init__.py ./app/__init__.py
COPY app/un_data_stream ./app/un_data_stream
COPY app/assets/*.csv ./app/assets/
COPY config ./config
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5"]
