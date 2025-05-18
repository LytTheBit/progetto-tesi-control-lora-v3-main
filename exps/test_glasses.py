import os, sys
import torch
from PIL import Image

# 1) Root del repo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 2) Import delle classi custom
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
from diffusers import UniPCMultistepScheduler

# 3) Carica l'UNet con canny
unet = UNet2DConditionModelEx.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
).add_extra_conditions(["canny"])

# 4) Pipeline ControlLoRA-v3
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# 5) Pesi LoRA “bicchieri”
lora_ckpt = os.path.join(
    project_root, "out", "lora-glasses", "checkpoint-4000", "pytorch_lora_weights.safetensors"
)
if not os.path.isfile(lora_ckpt):
    raise FileNotFoundError(f"LoRA checkpoint non trovato: {lora_ckpt}")
pipe.load_lora_weights(lora_ckpt)

# 6) Guida Canny e prompt per un bicchiere
# Sostituisci 'acqua-capri.png' con il file che vuoi testare
guide = Image.open(os.path.join(
    project_root, "glasses_data", "guide", "acqua-capri.png"
)).convert("RGB").resize((512, 512))

prompt = "Bicchiere in vetro trasparente, design essenziale con pareti sottili e linee pulite; vista frontale, sfondo bianco uniforme"

# 7) Generazione
out_dir = os.path.join(project_root, "out", "test_glasses")
os.makedirs(out_dir, exist_ok=True)
gen = torch.Generator(device="cuda").manual_seed(1234)

result = pipe(
    prompt=prompt,
    image=guide,
    num_inference_steps=80,
    guidance_scale=9.0,
    controlnet_conditioning_scale=1.2,
    generator=gen
)

# 8) Salvataggio
out_path = os.path.join(out_dir, "Bicchiere_test_5.png")
result.images[0].save(out_path)
print(f"Immagine salvata in: {out_path}")
