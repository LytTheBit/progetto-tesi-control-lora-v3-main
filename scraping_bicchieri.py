from icrawler.builtin import BingImageCrawler
import os

def download_glass_images(query, max_num=100, output_dir="dataset_bicchieri"):
    # Crea la cartella di destinazione se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Configura il crawler: 4 thread e cartella di storage
    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={"root_dir": output_dir}
    )
    # Avvia la ricerca e il download
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        # filtra per immagini grandi (opzionale)
        filters={"size": "large"}
    )


if __name__ == "__main__":
    queries = [
        "single wine glass on white background studio photo no watermark",
        "single clear drinking glass white background professional photo no watermark",
        "single stemless wine glass on white background high resolution photo no watermark",
        "single cocktail glass on white background product photo no watermark",
        "single whiskey glass white backdrop photo no watermark",
        "single beer glass on white background studio shot no watermark",
        "single champagne flute white background macro photo no watermark",
        "single glass tumbler on white background clean photo no watermark",
        "single stemmed wine glass white background product shot no watermark",
        "single glass mug on white background professional product photo no watermark"
    ]
    per_query = 25.0

    for q in queries:
        print(f"▶ Downloading '{q}'…")
        download_glass_images(q, max_num=per_query,
                              output_dir=f"dataset_bicchieri/{q[:20].replace(' ','_')}")
    print("✅ Fatto! Trovi i risultati in dataset_bicchieri/")

