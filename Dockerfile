FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge directement le modèle u2netp (~4,7 Mo) et vérifie qu'il n'est pas vide
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx', '/app/u2netp.onnx')" \
 && ls -l /app/u2netp.onnx \
 && test "$(stat -c%s /app/u2netp.onnx)" -gt 1000000

COPY app.py .
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1