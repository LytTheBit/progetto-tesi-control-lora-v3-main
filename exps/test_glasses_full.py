import os
import sys
import torch
import json
from PIL import Image
from itertools import product

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
    "SG161222/Realistic_Vision_V4.0_noVAE",
    subfolder="unet",
    torch_dtype=torch.float16
).add_extra_conditions(["canny"])

# 4) Pipeline ControlLoRA-v3
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

# 5) Pesi LoRA “bicchieri”
lora_ckpt = os.path.join(
    project_root, "out", "lora-glasses-test-realistic-vision-4-0",
    "lora-glasses-realistic-vision", "pytorch_lora_weights.safetensors"
)
if not os.path.isfile(lora_ckpt):
    raise FileNotFoundError(f"LoRA checkpoint non trovato: {lora_ckpt}")
pipe.load_lora_weights(lora_ckpt)

# 6) Guida Canny e prompt
guide_path = os.path.join(project_root, "glasses_data", "guide", "acqua-aurum.png")
guide = Image.open(guide_path).convert("RGB").resize((512, 512))

prompt = "transparent blue drinking glass with curved silhouette, isolated on white background, no shadows"
negative_prompt = "deformed, distorted, sketch, blurry, cartoon, colored background"
seed = 1234
gen = torch.Generator(device="cuda").manual_seed(seed)

# 7) Nuove variazioni
guidance_values = [5.0 + 5 * i for i in range(int((30.0 - 5.0) / 5) + 1)]  # [5,10,…,30]
steps_values    = list(range(150, 751, 75))                                 # [150,225,…,750]
cond_values     = [i * 0.1 for i in range(0, 11)]                           # [0.0,0.1,…,1.0]

# 8) Cartella unica di output
out_dir = os.path.join(project_root, "out", "tests-realistic-vision-4-0")
os.makedirs(out_dir, exist_ok=True)

# 9) Prepara contatore e raccolta settings
total = len(guidance_values) * len(steps_values) * len(cond_values)
counter = 0
all_settings = []

for guidance_scale, num_inference_steps, extra_condition_scale in product(
        guidance_values, steps_values, cond_values):

    counter += 1
    remaining = total - counter
    print(f"[{counter}/{total}] Generazione immagine: "
          f"guidance={guidance_scale:.1f}, steps={num_inference_steps}, "
          f"cond_scale={extra_condition_scale:.1f} (mancano {remaining})")

    # 10) Generazione
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=guide,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        extra_condition_scale=extra_condition_scale,
        generator=gen
    )

    # 11) Salvataggio immagine
    base = f"test-{guidance_scale:.0f}-{num_inference_steps}-{extra_condition_scale:.1f}"
    img_path = os.path.join(out_dir, f"{base}.png")
    result.images[0].save(img_path)
    #   print(f"✅ Salvato: {img_path}")

    # 12) Accumula impostazioni
    all_settings.append({
        "output_filename": f"{base}.png",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "condition_scale": extra_condition_scale,
        "training_checkpoint": lora_ckpt
    })

# 13) Salva in un unico JSON
settings_file = os.path.join(out_dir, "all_settings.json")
with open(settings_file, "w") as f:
    json.dump(all_settings, f, indent=4)
print(f"Tutte le impostazioni salvate in: {settings_file}")