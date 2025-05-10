import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, AutoencoderKL
from PIL import Image
import os

# 1) Carica il modello base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    revision="fp16"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# 2) Carica il LoRA “locale” (Unet)
lora_path = "out/lora-circles/adapter_5000.safetensors"
pipe.unet.load_attn_procs(lora_path)

# 3) Genera immagini di test
prompts = ["Cerchio rosso su sfondo bianco",
           "Cerchio blu su sfondo giallo",
           "Cerchio verde su sfondo grigio"]
os.makedirs("out/test_circles", exist_ok=True)

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device="cuda").manual_seed(42 + i)
    img = pipe(prompt,
               num_inference_steps=50,
               guidance_scale=7.5,
               generator=generator).images[0]
    out_path = f"out/test_circles/circle_{i:02d}.png"
    img.save(out_path)
    print("Saved", out_path)
