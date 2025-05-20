# Progetto Tesi
Progetto github universitario di Francesco Bonaiuti
Questo progetto varrà per il corso di PPM e la Tesi della triennale

Il progetto consiste nella creazione di un sito che faccia utilizzo di IA generativa per il design di bicchieri di diverso tipo

Il progetto usa come base il lavoro di HighCWu: https://github.com/HighCWu/control-lora-v3/tree/main/exps
![glassy-cups-pixel-art_505135-70](https://github.com/user-attachments/assets/3d547efd-7a05-40d2-9356-31f326834249)

Ecco i **file `.py`** presenti nel repo **LytTheBit/progetto-tesi-control-lora-v3-main** [GitHub](https://github.com/LytTheBit/progetto-tesi-control-lora-v3-main):

- **`model.py`**  
    Contiene la classe `UNet2DConditionModelEx`, un’estensione del modello UNet di Stable Diffusion che aggiunge canali extra di condizionamento (es. mappe Canny).
    
- **`pipeline.py`**  
    Definisce la pipeline `StableDiffusionControlLoraV3Pipeline`, che unisce ControlNet e LoRA su Stable Diffusion v1.5 per fare generazione condizionata.
    
- **`pipeline_sdxl.py`**  
    Analogo a `pipeline.py` ma progettato per i modelli SDXL, con scheduler e formati specifici.
    
- **`train.py`**  
    Script di training per ControlLoRA-v3 su Stable Diffusion v1.5. Gestisce:
    
    - parsing degli argomenti CLI,
        
    - caricamento del dataset (tramite `sd1_5_tile_pair_data.py` nella cartella `exps/`),
        
    - ciclo di ottimizzazione,
        
    - salvataggio checkpoint,
        
    - logging.
        
- **`train_sdxl.py`**  
    Versione di `train.py` adattata per Stable Diffusion XL.
    
- **`requirements.txt`**  
    Elenco delle dipendenze Python richieste dal progetto.
    
- **Cartella `exps/`**  
    Contiene il **dataset loader** custom `sd1_5_tile_pair_data.py` (la classe `TrainDataset` che carica il CSV di caption e le due cartelle `image/`/`guide/`, applica trasformazioni e prepara i batch).
