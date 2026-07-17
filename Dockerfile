# ServerGuard AI - Predictive Server Failure Detection
# Containerized Flask dashboard

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy the full project
COPY . .

# The dashboard reads models/ and data/processed/ at import time,
# so those must exist before the container starts. This assumes
# the ML pipeline (generate_data.py -> train_models.py -> prepare_dashboard_data.py)
# was already run and its outputs are included in the build context.

EXPOSE 5000

WORKDIR /app/app

# Use gunicorn instead of Flask's dev server for anything beyond local testing
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
