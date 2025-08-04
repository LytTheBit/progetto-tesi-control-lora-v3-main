#!/usr/bin/env python3
import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

BASE      = "https://www.borselliniforniturealberghiere.it"
START_URL = BASE + "/112-vetri-e-bicchieri"
DEST_DIR  = "bicchieri_images"
MIN_SIZE  = 400   # scarta tutte le immagini più piccole di 400x400

# headers di cortesia
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(DEST_DIR, exist_ok=True)

def is_same_domain(url):
    return urlparse(url).netloc == urlparse(BASE).netloc

def fetch_html(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.text

def get_image_urls_from(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    imgs = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        full = urljoin(page_url, src.split("?")[0])
        if full.lower().endswith((".png",".jpg",".jpeg")):
            imgs.add(full)
    return imgs

def get_links_from(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(page_url, a["href"].split("?")[0])
        if is_same_domain(full):
            links.add(full)
    return links

def download_and_filter(img_url):
    try:
        resp = requests.get(img_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        im = Image.open(BytesIO(resp.content))
        w,h = im.size
        if w>=MIN_SIZE and h>=MIN_SIZE:
            fname = os.path.basename(urlparse(img_url).path)
            path  = os.path.join(DEST_DIR, fname)
            with open(path, "wb") as f:
                f.write(resp.content)
            print("⤓", fname, f"({w}×{h})")
    except Exception as e:
        print("✗", img_url, e)

def crawl(start_url, max_depth=2):
    seen_pages = set([start_url])
    pages      = [(start_url, 0)]
    while pages:
        url, depth = pages.pop(0)
        print(f"\n→ Scansiono {url} (depth {depth})")
        html = fetch_html(url)
        # 1) scarica tutte le immagini "grandi"
        for img in get_image_urls_from(html, url):
            download_and_filter(img)
        # 2) se non ho superato la profondità, segui tutti i link interni
        if depth < max_depth:
            for link in get_links_from(html, url):
                if link not in seen_pages:
                    seen_pages.add(link)
                    pages.append((link, depth+1))

if __name__ == "__main__":
    crawl(START_URL)

