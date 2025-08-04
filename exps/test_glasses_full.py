import os, sys
import torch
import json
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

# 5) Pesi LoRA “bicchieri”
lora_ckpt = os.path.join(
    project_root, "out", "lora-glasses-test-realistic-vision-4-0",
    "checkpoint-3000", "pytorch_lora_weights.safetensors"
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

# 7) Imposta le variazioni
guidance_values = [5.0 + 2.5 * i for i in range(int((30.0 - 5.0) / 2.5) + 1)]
steps_values    = list(range(150, 801, 50))
cond_values     = [i * 0.1 for i in range(0, 16)]  # da 0.0 a 1.5 passo 0.1

# 8) Cartella unica di output
out_dir = os.path.join(project_root, "out", "all_tests")
os.makedirs(out_dir, exist_ok=True)

for guidance_scale in guidance_values:
    for num_inference_steps in steps_values:
        for extra_condition_scale in cond_values:
            # nome base senza decimali superflui
            base = f"test-{guidance_scale:.1f}-{num_inference_steps}-{extra_condition_scale:.1f}"
            img_path = os.path.join(out_dir, f"{base}.png")
            cfg_path = os.path.join(out_dir, f"{base}_settings.json")

            # 9) Generazione
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=guide,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                extra_condition_scale=extra_condition_scale,
                generator=gen
            )

            # 10) Salvataggio immagine
            result.images[0].save(img_path)

            # 11) Salvataggio impostazioni
            settings = {
                "output_filename": f"{base}.png",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guide_image": guide_path,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "condition_scale": extra_condition_scale,
                "seed": seed,
                "training_checkpoint": lora_ckpt
            }
            with open(cfg_path, "w") as f:
                json.dump(settings, f, indent=4)

            print(f"✅ Salvato: {img_path}")
