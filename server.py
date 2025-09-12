import os
import io
import base64
import torch
from fastapi import FastAPI, Request, HTTPException
from PIL import Image

import asyncio, time, uuid, contextlib
from typing import Any, Dict, Optional

# Config runtime (override con env)
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", "32"))     # cap coda
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "1"))  # worker attivi (GPU=1)
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "1200"))     # 20 min job
WAIT_TIMEOUT_SEC = int(os.environ.get("WAIT_TIMEOUT_SEC", "900"))    # 15 min attesa endpoint


# 1. Imposta il path corretto e carica i moduli custom
project_root = os.path.abspath(os.path.dirname(__file__))
os.chdir(project_root)
import sys
sys.path.insert(0, project_root)

# Cartella MEDIA del sito (di default punta al progetto Django in locale)
LORA_DIR = os.environ.get(
    "LORA_DIR",
    os.path.normpath(os.path.join(project_root, "..", "Design_maker_online", "media"))
)

# 2. Import pipeline IA
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
from diffusers import UniPCMultistepScheduler

# 3. Inizializza la pipeline (senza weights)
unet = (
    UNet2DConditionModelEx
    .from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet", torch_dtype=torch.float16)
    .add_extra_conditions(["canny"])
)
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# (Opzionale) carica un LoRA di default all'avvio, per non lasciare la pipeline "vuota"
default_ckpt = os.path.join(project_root, "modelli", "lora-glasses-base", "pytorch_lora_weights.safetensors")
if os.path.isfile(default_ckpt):
    pipe.load_lora_weights(default_ckpt)

# 4. FastAPI app
app = FastAPI()

# 5. Job queue e worker per serializzare le richieste
class Job:
    __slots__ = ("id", "data", "future", "enqueued_at")
    def __init__(self, data: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.data = data
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.enqueued_at = time.time()

job_queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
jobs_registry: Dict[str, Dict[str, Any]] = {}

# Funzione di generazione IA
async def worker_loop(worker_id: int):
    while True:
        job: Job = await job_queue.get()
        jobs_registry[job.id] = {"status": "running", "enqueued_at": job.enqueued_at, "started_at": time.time()}
        try:
            async with asyncio.timeout(JOB_TIMEOUT_SEC):
                result = await run_generation(job.data)
                jobs_registry[job.id].update({"status": "done", "ended_at": time.time()})
                if not job.future.done():
                    job.future.set_result({"image": result, "job_id": job.id})
        except asyncio.TimeoutError:
            jobs_registry[job.id].update({"status": "timeout", "ended_at": time.time()})
            if not job.future.done():
                job.future.set_exception(HTTPException(status_code=504, detail="Timeout generazione"))
        except Exception as e:
            jobs_registry[job.id].update({"status": "error", "ended_at": time.time(), "error": str(e)})
            if not job.future.done():
                job.future.set_exception(HTTPException(status_code=500, detail=f"Errore generazione: {e}"))
        finally:
            job_queue.task_done()

# Wrapper per eseguire la generazione in un thread separato
@app.on_event("startup")
async def _startup():
    app.state.workers = [asyncio.create_task(worker_loop(i)) for i in range(WORKER_CONCURRENCY)]

# Shutdown pulito
@app.on_event("shutdown")
async def _shutdown():
    for t in getattr(app.state, "workers", []):
        t.cancel()
    for t in getattr(app.state, "workers", []):
        with contextlib.suppress(Exception):
            await t

# Funzione di generazione sincrona (da eseguire in thread separato)
def _run_generation_sync(data: Dict[str, Any]) -> str:
    # === COPIATA la tua logica dall’endpoint (validazioni incluse) ===

    # --- Selezione LoRA ---
    rel_path = data.get("model_file")
    if not rel_path:
        model_name = data.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Parametro 'model' o 'model_file' mancante")
        rel_path = f"lora/{model_name}.safetensors"

    lora_path = os.path.normpath(os.path.join(LORA_DIR, rel_path))
    if not os.path.isfile(lora_path):
        raise HTTPException(status_code=400, detail=f"LoRA non trovato: {lora_path}")

    # Reset + carico LoRA
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "attn_processors"):
        pipe.unload_lora_weights()
    try:
        pipe.load_lora_weights(lora_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel caricamento LoRA: {e}")

    # --- Prompt ---
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt mancante")
    negative_prompt = data.get(
        "negative_prompt",
        "cartoon, sketch, distorted, colored background, reflections, ornate, decorative glass, textured sides, blurry, surreal"
    ).strip()

    # --- Canny ---
    canny_data = data.get("canny", "")
    if not isinstance(canny_data, str) or not canny_data.startswith("data:image"):
        raise HTTPException(status_code=400, detail="Formato Canny non valido")
    try:
        _, body = canny_data.split(",", 1)
        canny_bytes = base64.b64decode(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Errore decodifica Canny")
    guide = Image.open(io.BytesIO(canny_bytes)).convert("RGB").resize((512, 512))

    # --- Parametri ---
    try:
        steps    = int( data.get("num_inference_steps", 150) )
        guidance = float( data.get("guidance_scale", 20) )
        extra    = float( data.get("extra_condition_scale", 0.6) )
    except ValueError:
        raise HTTPException(status_code=400, detail="Parametri numerici non validi")

    gen = torch.Generator(device="cuda").manual_seed(1234)

    # --- Generazione ---
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=guide,
            num_inference_steps=steps,
            guidance_scale=guidance,
            extra_condition_scale=extra,
            generator=gen
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore generazione IA: {e}")

    # --- PNG base64 ---
    output_io = io.BytesIO()
    result.images[0].save(output_io, format="PNG")
    img_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

async def run_generation(data: Dict[str, Any]) -> str:
    # Sposta il lavoro bloccante fuori dall’event loop
    return await asyncio.to_thread(_run_generation_sync, data)


# Endpoint per controllare lo stato di un job
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    if job_queue.full():
        raise HTTPException(status_code=429, detail="Coda piena, riprova più tardi")

    job = Job(data)
    jobs_registry[job.id] = {"status": "queued", "enqueued_at": job.enqueued_at}
    await job_queue.put(job)

    try:
        async with asyncio.timeout(WAIT_TIMEOUT_SEC):
            result = await job.future  # {"image": ..., "job_id": ...}
            return result
    except asyncio.TimeoutError:
        # Job ancora in corso: fai polling col job_id
        return {"job_id": job.id, "status": "running", "detail": "Controlla lo stato su /jobs/{id}"}


# Endpoint per controllare lo stato di un job
@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    info = jobs_registry.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return {"job_id": job_id, **info}

# Endpoint di health check
@app.get("/health")
async def health():
    return {
        "queue_size": job_queue.qsize(),
        "queue_max": QUEUE_MAXSIZE,
        "running": sum(1 for j in jobs_registry.values() if j.get("status") == "running"),
        "queued": sum(1 for j in jobs_registry.values() if j.get("status") == "queued"),
    }
