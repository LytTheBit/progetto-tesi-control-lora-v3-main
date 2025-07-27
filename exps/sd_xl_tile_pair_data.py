import os
import sys
import copy
import torch
import random
import numpy as np

from PIL import Image, ImageFilter
from datasets import load_dataset, DatasetDict
from torchvision import transforms
from accelerate.logging import get_logger

logger = get_logger(__name__)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizers, accelerator):
        # 1) Percorso base del dataset
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../glasses_data")
        )
        self.args = copy.deepcopy(args)
        self.tokenizer = tokenizers[0]  # primo tokenizer è quello di testo
        self.accelerator = accelerator

        # 2) Caricamento CSV con le didascalie
        data_files = {"train": os.path.join(self.data_dir, "captions.csv")}
        dataset = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=self.args.cache_dir
        )

        # 3) Aggiunta dei path completi per 'image' e 'guide'
        def add_paths(ex):
            ex["image"] = os.path.join(self.data_dir, "image", ex["file"])
            ex["guide"] = os.path.join(self.data_dir, "guide", ex["file"])
            return ex

        ds = dataset["train"].map(add_paths, remove_columns=[])
        ds = DatasetDict({"train": ds})

        column_names = ds["train"].column_names

        # Colonna immagine
        self.image_column = args.image_column or column_names[0]
        if self.image_column not in column_names:
            raise ValueError(f"--image_column '{self.image_column}' non trovata in {column_names}")

        # Colonna didascalia
        self.caption_column = args.caption_column or column_names[1]
        if self.caption_column not in column_names:
            raise ValueError(f"--caption_column '{self.caption_column}' non trovata in {column_names}")

        # Colonna immagine di conditioning (guide)
        self.conditioning_image_column = args.conditioning_image_column or "guide"
        if self.conditioning_image_column not in column_names:
            raise ValueError(f"--conditioning_image_column '{self.conditioning_image_column}' non trovata in {column_names}")

        # Trasformazioni per le immagini
        self.image_transforms = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Sotto-campionamento (opzionale)
        if args.max_train_samples:
            ds["train"] = ds["train"].shuffle(seed=args.seed).select(
                range(min(len(ds["train"]), args.max_train_samples))
            )

        # Applichiamo la funzione di preprocessing
        train_ds = ds["train"].with_transform(self.preprocess_train)
        self.dataset = train_ds

    def tokenize_captions(self, examples):
        captions = []
        for cap in examples[self.caption_column]:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(cap, str):
                captions.append(cap)
            else:
                captions.append(str(cap))
        enc = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc.input_ids

    def preprocess_train(self, examples):
        # Carica PIL images (qui p è già il path completo da add_paths)
        examples[self.image_column] = [
            Image.open(p).convert("RGB")
            for p in examples[self.image_column]
        ]
        examples[self.conditioning_image_column] = [
            Image.open(p).convert("RGB")
            for p in examples[self.conditioning_image_column]
        ]

        # Trasformazioni
        pix = [self.image_transforms(img) for img in examples[self.image_column]]
        cond = [self.conditioning_image_transforms(img) for img in examples[self.conditioning_image_column]]
        examples["pixel_values"] = pix
        examples["conditioning_pixel_values"] = cond

        # Tokenizza le didascalie
        examples["input_ids"] = self.tokenize_captions(examples)
        return examples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]