import io
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image
from rembg import remove, new_session

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Petit modèle (~4 Mo), adapté aux 512 Mo de RAM du palier gratuit
session = new_session("u2netp")

MAX_SIDE = 768  # réduit l'image pour économiser la RAM

def process(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = remove(img, session=session)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
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