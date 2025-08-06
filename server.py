import os
import io
import base64
import torch
from fastapi import FastAPI, Request, HTTPException
from PIL import Image

# 1. Imposta il path corretto e carica i moduli custom
project_root = os.path.abspath(os.path.dirname(__file__))
os.chdir(project_root)
import sys
sys.path.insert(0, project_root)

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

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    # --- Estrai modello e carica il relativo LoRA ---
    model_name = data.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Parametro 'model' mancante")
    lora_path = os.path.join(project_root, "modelli", model_name, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_path):
        raise HTTPException(status_code=400, detail=f"Modello '{model_name}' non trovato in modelli/")
    # Rimuove adapter già presenti (necessario per evitare errori "already in use")
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "attn_processors"):
        pipe.unload_lora_weights()  # Reset completo dei LoRA

    pipe.load_lora_weights(lora_path, adapter_name="current")  # Puoi usare sempre lo stesso nome

    # --- Prompts ---
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt mancante")
    negative_prompt = data.get(
        "negative_prompt",
        "cartoon, sketch, distorted, colored background, reflections, ornate, decorative glass, textured sides, blurry, surreal"
    ).strip()

    # --- Decodifica Canny ---
    canny_data = data.get("canny", "")
    if not isinstance(canny_data, str) or not canny_data.startswith("data:image"):
        raise HTTPException(status_code=400, detail="Formato Canny non valido")
    try:
        _, body = canny_data.split(",", 1)
        canny_bytes = base64.b64decode(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Errore decodifica Canny")
    guide = Image.open(io.BytesIO(canny_bytes)).convert("RGB").resize((512, 512))

    # --- Parametri numerici ---
    try:
        steps        = int( data.get("num_inference_steps", 80) )
        guidance     = float( data.get("guidance_scale",        9.0) )
        extra_scale  = float( data.get("extra_condition_scale",  1.2) )
    except ValueError:
        raise HTTPException(status_code=400, detail="Parametri numerici non validi")

    # --- Seed per riproducibilità ---
    gen = torch.Generator(device="cuda").manual_seed(1234)

    # --- Generazione immagine ---
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=guide,
            num_inference_steps=steps,
            guidance_scale=guidance,
            extra_condition_scale=extra_scale,
            generator=gen
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore generazione IA: {e}")

    # --- Ritorna PNG in base64 ---
    output_io = io.BytesIO()
    result.images[0].save(output_io, format="PNG")
    img_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")
    return {"image": f"data:image/png;base64,{img_base64}"}