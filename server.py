# server.py
import os
import io
import base64
import torch
from fastapi import FastAPI, Request, HTTPException
from PIL import Image

# 1. Imposta il path corretto per importare il codice esistente
project_root = os.path.abspath(os.path.dirname(__file__))
os.chdir(project_root)
import sys
sys.path.insert(0, project_root)

# 2. Importa pipeline IA
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
from diffusers import UniPCMultistepScheduler

# 3. Setup pipeline come in test_glasses.py
unet = UNet2DConditionModelEx.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
).add_extra_conditions(["canny"])

pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

lora_ckpt = os.path.join(
    project_root,
    "out", "lora-glasses", "checkpoint-5000", "pytorch_lora_weights.safetensors"
)
pipe.load_lora_weights(lora_ckpt)

# 4. FastAPI app
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    # Estrai i prompt
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt mancante")

    negative_prompt = data.get(
        "negative_prompt",
        "cartoon, sketch, distorted, colored background, reflections, ornate, decorative glass, textured sides, blurry, surreal"
    ).strip()

    # Estrai e decodifica il Canny
    canny_data = data.get("canny", "")
    if not isinstance(canny_data, str) or not canny_data.startswith("data:image"):
        raise HTTPException(status_code=400, detail="Formato Canny non valido")
    try:
        header, body = canny_data.split(",", 1)
        canny_bytes = base64.b64decode(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Errore decodifica Canny")

    # Prepara l'immagine per ControlNet
    guide = Image.open(io.BytesIO(canny_bytes)).convert("RGB").resize((512, 512))

    # Parametri numerici con default
    try:
        steps      = int( data.get("num_inference_steps", 80) )
        guidance   = float( data.get("guidance_scale",         9.0) )
        cond_scale = float( data.get("controlnet_conditioning_scale", 1.2) )
    except ValueError:
        raise HTTPException(status_code=400, detail="Parametri numerici non validi")

    # Seed per riproducibilità
    gen = torch.Generator(device="cuda").manual_seed(1234)

    # Esecuzione pipeline
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=guide,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=cond_scale,
            generator=gen
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore generazione IA: {e}")

    # Codifica il risultato in base64
    output_io = io.BytesIO()
    result.images[0].save(output_io, format="PNG")
    img_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_base64}"}

