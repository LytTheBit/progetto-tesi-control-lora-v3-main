import os
import sys
import torch
from diffusers import AutoencoderKL, UniPCMultistepScheduler, StableDiffusionPipeline
from PIL import Image

# Imposta la cartella principale del progetto
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)

# Identificativo del modello
model_id = "SG161222/Realistic_Vision_V4.0"

# Carica il VAE in fp16 su GPU
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
)

# Carica la pipeline base di Stable Diffusion con il VAE custom
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

# Sostituisci lo scheduler col più efficiente UniPCMultistep
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Prompt per generare immagini a colori
prompts = [
    "portrait of a young black superhero girl with short blonde Afro, sexy red lips, tall and slim, Brooklyn background, highly detailed gold jewelry, digital art, intricate, sharp focus, trending on Artstation, unreal engine 5, 4K UHD image, by Brom and Artgerm, face by Otto Schmidt",
    # aggiungi altri prompt se vuoi
]

# Directory di output
dir_out = "./out/full"
os.makedirs(dir_out, exist_ok=True)

# Parametri di generazione
guidance_scale = 7.5
num_inference_steps = 50

# Loop di generazione
for idx, prompt in enumerate(prompts):
    generator = torch.Generator(device="cuda").manual_seed(42)
    result = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )
    image = result.images[0]
    image.save(os.path.join(dir_out, f"full_{idx}.png"))
    print(f"Saved full-color image: full_{idx}.png")
