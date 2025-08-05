import os, sys
import torch
import json
from PIL import Image
from datetime import datetime

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
    #"stabilityai/stable-diffusion-2-1-base",
    "SG161222/Realistic_Vision_V4.0_noVAE",
    subfolder="unet",
    torch_dtype=torch.float16
).add_extra_conditions(["canny"])

# 4) Pipeline ControlLoRA-v3
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    #"stabilityai/stable-diffusion-2-1-base",
    "SG161222/Realistic_Vision_V4.0_noVAE",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# 5) Pesi LoRA “bicchieri”
lora_ckpt = os.path.join(
    #project_root, "out", "lora-glasses-test-stable-diffusion-2-1", "lora-glasses-realistic-vision", "pytorch_lora_weights.safetensors"
    project_root, "out", "lora-glasses-test-realistic-vision-4-0", "lora-glasses-realistic-vision", "pytorch_lora_weights.safetensors"
)
if not os.path.isfile(lora_ckpt):
    raise FileNotFoundError(f"LoRA checkpoint non trovato: {lora_ckpt}")
pipe.load_lora_weights(lora_ckpt)

# 6) Guida Canny e prompt per un bicchiere
guide_path = os.path.join(project_root, "glasses_data", "guide", "acqua-aurum.png")
guide = Image.open(guide_path).convert("RGB").resize((512, 512))

prompt = "transparent blue drinking glass with curved silhouette, isolated on white background, no shadows"
negative_prompt = "deformed, distorted, sketch, blurry, cartoon, colored background"
seed = 1234
num_inference_steps = 750 # Numero di step della catena di denoising. Più è alto, più a lungo il modello "pulisce" l'immagine
guidance_scale = 25.0 # Bilancia il peso tra il prompt testuale e la creatività del modello
extra_condition_scale = 1.3 # Controlla quanto peso ha l’immagine guida fornita al ControlNet.


# 7) Generazione
out_dir = os.path.join(project_root, "out", "test_glasses-RV")
os.makedirs(out_dir, exist_ok=True)
gen = torch.Generator(device="cuda").manual_seed(seed)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=guide,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    extra_condition_scale=extra_condition_scale,
    generator=gen
)

# 8) Salvataggio immagine + impostazioni con contatore incrementale

def get_next_index(directory, prefix="bicchiere_", extension=".png"):
    existing = [
        fname for fname in os.listdir(directory)
        if fname.startswith(prefix) and fname.endswith(extension)
    ]
    numbers = []
    for fname in existing:
        try:
            number = int(fname.replace(prefix, "").replace(extension, ""))
            numbers.append(number)
        except ValueError:
            continue
    next_index = max(numbers, default=0) + 1
    return f"{prefix}{next_index:04d}"

# Genera nome univoco incrementale
basename = get_next_index(out_dir)
image_path = os.path.join(out_dir, f"{basename}.png")
settings_path = os.path.join(out_dir, f"{basename}_settings.json")

# Salva immagine
result.images[0].save(image_path)

# Salva impostazioni
settings = {
    "output_filename": f"{basename}.png",
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "guide_image": guide_path,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
    "condition_scale": extra_condition_scale,
    "seed": seed,
    "training_checkpoint": lora_ckpt
}
with open(settings_path, "w") as f:
    json.dump(settings, f, indent=4)

print(f"Immagine salvata in: {image_path}")
print(f"Impostazioni salvate in: {settings_path}")
