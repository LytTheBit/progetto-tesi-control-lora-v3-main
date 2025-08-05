import os, sys
import torch
from PIL import Image

# 1) Inserisci la root del repo nel PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 2) Import delle classi custom
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
from diffusers import UniPCMultistepScheduler

# 3) Carica l'UNet esteso con il canale "canny"
unet = UNet2DConditionModelEx.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
)
# aggiunge il condizionamento canny
unet = unet.add_extra_conditions(["canny"])

# 4) Costruisci la pipeline ControlLoRA-v3 passando l'unet già pronto
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None
)
# sostituisci lo scheduler se vuoi
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# 5) Carica i pesi LoRA dal lora-glasses-base
lora_ckpt = os.path.join(
    project_root, "out", "lora-circles", "lora-glasses-base", "pytorch_lora_weights.safetensors"
)
if not os.path.isfile(lora_ckpt):
    raise FileNotFoundError(f"LoRA checkpoint non trovato: {lora_ckpt}")
pipe.load_lora_weights(lora_ckpt)

# 6) Prepara la guida (Canny) e il prompt
guide = Image.open(os.path.join(project_root, "circle_data", "guide", "circle_000.png")) \
              .convert("RGB") \
              .resize((512, 512))
prompt = "Cerchio rosso su sfondo bianco"

# 7) Generazione
os.makedirs(os.path.join(project_root, "out", "test_circles"), exist_ok=True)
gen = torch.Generator(device="cuda").manual_seed(42)
result = pipe(
    prompt=prompt,
    image=guide,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=gen
)

# 8) Salva l'immagine
out_path = os.path.join(project_root, "out", "test_circles", "circle_test.png")
result.images[0].save(out_path)
print(f"Immagine salvata in: {out_path}")