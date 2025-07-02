FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt ./
COPY pyproject.toml ./

# Install Python dependencies (including local package in editable mode)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .


# COPY artifacts/training/model.h5 /app/artifacts/training/model.h5

# Copy the rest of the code
COPY . /app


EXPOSE 5000

CMD ["python3", "app.py"]