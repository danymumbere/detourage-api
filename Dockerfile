FROM python:3.11-slim AS model
RUN pip install --no-cache-dir "rembg[cpu]"
RUN python -c "from rembg import new_session; new_session('u2netp')"

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=model /root/.u2net/u2netp.onnx /app/u2netp.onnx
COPY app.py .
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1