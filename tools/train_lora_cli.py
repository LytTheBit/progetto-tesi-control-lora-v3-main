#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer LoRA minimale per SD 1.5 con Diffusers.
- Legge immagini + captions (da cartella dataset_dir)
- Allena solo UNet con LoRA (rank configurabile)
- Stampa "step=<N>" ad ogni iterazione -> usato per la progress bar
- Salva "pytorch_lora_weights.safetensors" in out_dir
ATTENZIONE: è un trainer minimale, utile come base. Per produzione
aggiungi: gradient_accumulation, mixed precision, checkpoints, ecc.
"""
import argparse, os, math, random, glob, json, time, sys
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0, AttnProcessor2_0
from transformers import CLIPTokenizer

def list_images(root):
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def load_captions(captions_path):
    m = {}
    if captions_path and Path(captions_path).exists():
        for line in Path(captions_path).read_text(encoding="utf-8").splitlines():
            if "|" in line:
                name, cap = line.split("|", 1)
                m[name.strip()] = cap.strip()
    return m

class ImageTextDataset(Dataset):
    def __init__(self, root, tokenizer: CLIPTokenizer, size=512):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.size = size
        self.items = list_images(root)
        self.capmap = {}
        # captions.txt può essere nel root del dataset
        capfile = self.root / "captions.txt"
        if capfile.exists():
            self.capmap = load_captions(capfile)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        # caption: cerca prima nome file esatto, poi nome base
        name = p.name
        caption = self.capmap.get(name) or self.capmap.get(str(p.relative_to(self.root)).replace("\\","/")) or ""
        try:
            image = Image.open(p).convert("RGB")
        except Exception:
            # se un file è corrotto, salta
            return self[(idx + 1) % len(self)]
        # resize + center crop a 512x512
        w, h = image.size
        s = min(w, h)
        image = image.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((self.size, self.size), Image.BICUBIC)
        # to tensor in [-1,1]
        x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
                              .float().view(self.size, self.size, 3) / 255.0).numpy()).permute(2,0,1)
        x = (x * 2.0) - 1.0
        ids = self.tokenizer(
            caption if caption else "",
            padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        return {"pixel_values": x, "input_ids": ids}

def add_lora_to_unet(unet: nn.Module, rank: int):
    # Sostituisce tutti i processor di attention con LoRA
    loras = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_processor"):
            proc = module.processor
            hidden_size = None
            try:
                hidden_size = module.to_q.in_features
            except Exception:
                pass
            if isinstance(proc, AttnProcessor2_0):
                lora = LoRAAttnProcessor2_0(hidden_size=hidden_size, rank=rank)
            else:
                lora = LoRAAttnProcessor(hidden_size=hidden_size, rank=rank)
            module.set_processor(lora)
            loras.append(lora)
    # restituisci parametri allenabili
    params = []
    for lp in loras:
        params += list(lp.parameters())
    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id o path locale (es. runwayml/stable-diffusion-v1-5)")
    ap.add_argument("--dataset", required=True, help="cartella con immagini e (opzionale) captions.txt")
    ap.add_argument("--out", required=True, help="cartella di output")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Carica pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")
    pipe.to(device)

    # tokenizer per le captions
    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")

    # Dataset & loader
    ds = ImageTextDataset(args.dataset, tokenizer, size=512)
    if len(ds) == 0:
        print("Nessuna immagine trovata nel dataset.", file=sys.stderr, flush=True)
        sys.exit(2)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # Abilita LoRA sull'UNet
    trainable_params = add_lora_to_unet(pipe.unet, rank=args.rank)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Training loop minimale
    global_step = 0
    pipe.unet.train()
    pipe.text_encoder.eval()  # allena solo UNet
    for step in range(args.steps):
        batch = next(iter(dl))
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]
            # prepara latenti dal VAE
            images = batch["pixel_values"].to(device, dtype=pipe.unet.dtype)
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
        # aggiungi rumore
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # predici il rumore
        pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = nn.functional.mse_loss(pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        global_step += 1
        print(f"step={global_step}", flush=True)

    # Salvataggio compatibile 0.25.x (AttnProcessor) e 0.26+ (LoRA layers)
    try:
        # diffusers >= 0.26 (LoRA layers)
        pipe.save_lora_weights(args.out, weight_name="pytorch_lora_weights.safetensors")
    except Exception:
        # diffusers 0.25.x (AttnProcessor)
        pipe.save_attn_procs(args.out, safe_serialization=True)
        # normalizza il nome del file a quello atteso dal resto del progetto
        import os, glob
        target = os.path.join(args.out, "pytorch_lora_weights.safetensors")
        if not os.path.exists(target):
            cand = glob.glob(os.path.join(args.out, "*.safetensors"))
            if cand:
                os.replace(cand[0], target)

    print("done", flush=True)

if __name__ == "__main__":
    main()