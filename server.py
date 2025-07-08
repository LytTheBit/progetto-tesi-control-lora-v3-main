# server.py
import os
import io
import base64
import torch
from fastapi import FastAPI, Request
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
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
).add_extra_conditions(["canny"])

pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

lora_ckpt = os.path.join(project_root, "out", "lora-glasses", "checkpoint-5000", "pytorch_lora_weights.safetensors")
pipe.load_lora_weights(lora_ckpt)

# 4. FastAPI
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    negative_prompt = "cartoon, sketch, distorted, colored background, reflections, ornate, decorative glass, textured sides, blurry, surreal" # prompo negativi non modificabili dal utente
    canny_data = base64.b64decode(data.get("canny").split(",")[1])
    guide = Image.open(io.BytesIO(canny_data)).convert("RGB").resize((512, 512))
    gen = torch.Generator(device="cuda").manual_seed(1234)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt, # prompo negativi non modificabili dal utente
        image=guide,
        num_inference_steps=int(data.get("num_inference_steps", 80)),
        guidance_scale=float(data.get("guidance_scale", 9.0)),
        controlnet_conditioning_scale=float(data.get("controlnet_conditioning_scale", 1.2)),
        generator=gen
    )


    # Codifica il risultato in base64
    output_io = io.BytesIO()
    result.images[0].save(output_io, format="PNG")
    img_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")
    return {"image": f"data:image/png;base64,{img_base64}"}
