FROM python:3.10-slim

# System dependencies (curl needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 portaudio19-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose default port (Cloud Run overrides this via $PORT)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run Streamlit with dynamic port and headless mode (critical for Cloud Run)
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
