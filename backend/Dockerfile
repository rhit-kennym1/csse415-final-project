FROM python:3.11-slim

# Build deps for lxml, xgboost, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend package (sibling files set the working module path)
COPY . ./backend/
# Trained models live under backend/models/ — must exist locally before docker build
RUN test -f backend/models/gb.joblib || (echo "Missing models — run python -m backend.train_models first" && exit 1)

ENV PORT=8080
EXPOSE 8080

# Cloud Run sets $PORT — uvicorn binds to it.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
