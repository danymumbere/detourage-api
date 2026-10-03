import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIDE = 640

# Réglages qui limitent la consommation de mémoire
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False

session = ort.InferenceSession(
    "u2netp.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def process(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    # Prétraitement attendu par U2-Net : 320x320, normalisé
    small = img.resize((320, 320), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

    pred = session.run(None, {input_name: arr})[0][0, 0]
    mn, mx = float(pred.min()), float(pred.max())
    pred = (pred - mn) / max(mx - mn, 1e-6)

    # Masque remis à la taille de l'image, utilisé comme canal de transparence
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.LANCZOS)
    img.putalpha(mask)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/remove")
async def remove_bg(request: Request):
    data = await request.body()
    if not data:
        return Response("Image vide", status_code=400)
    try:
        png = await run_in_threadpool(process, data)
    except Exception as e:
        return Response(f"Erreur : {e}", status_code=500)
    return Response(content=png, media_type="image/png")