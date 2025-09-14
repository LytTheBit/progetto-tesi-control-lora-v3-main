import argparse, os, time, sys, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Simula training: stampa "step=.." così la barra sale
    for i in range(1, args.steps + 1):
        print(f"step={i}", flush=True)
        time.sleep(0.02)  # accelera o rallenta per test

    # Crea un finto file .safetensors per chiudere il flusso
    path = pathlib.Path(args.out) / "dummy.safetensors"
    with open(path, "wb") as f:
        f.write(b"FAKE-LORA")

    print("done", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
