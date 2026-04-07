FROM python:3.12.13-slim-trixie

WORKDIR /app

# System deps (important for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "clearview_backend_single_file:app", "--host", "0.0.0.0", "--port", "8000"]