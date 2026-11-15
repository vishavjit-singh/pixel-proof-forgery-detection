# API image. Trains the model at build time on the synthetic benchmark (a few minutes on CPU),
# so the container starts ready to serve.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
RUN python -m pixelproof generate && python -m pixelproof train --export-patches "" && rm -rf data

EXPOSE 8000
CMD ["python", "-m", "pixelproof", "serve", "--host", "0.0.0.0", "--port", "8000"]
