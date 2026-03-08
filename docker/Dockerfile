# Hull Tactical Market Prediction - Streamlit App
FROM python:3.12-slim

LABEL maintainer="Hull Tactical Team"
LABEL description="Hull Tactical Market Prediction — Streamlit Dashboard"

WORKDIR /app

# curl required for health check; build-essential for C-extension packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY app/ app/
COPY artifacts/ artifacts/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
