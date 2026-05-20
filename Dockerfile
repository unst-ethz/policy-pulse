FROM python:3.11-slim

WORKDIR /app

# To be set by the CI
ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
ENV BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_DATE=${BUILD_DATE}

# Set to True at runtime if we want to show that it's experimental
ENV WARN_EXPERIMENTAL=False

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8050

# Preload the application let's a masterprocess fetch and process our data before spawning workers for handling requests.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "300", "--preload", "app.__main__:server"]
