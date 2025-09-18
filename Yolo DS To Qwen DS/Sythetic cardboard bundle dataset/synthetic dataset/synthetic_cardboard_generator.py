#!/usr/bin/env python3
"""
Create a dataset by EDITING from a real factory photo using gpt-image-1.

• Uses a very short, low-restriction prompt that keeps the photo’s realism.
• Randomizes strap color, label count, and 2-by-2 bundle shade pairs.
• Optional mask: transparent regions indicate where edits are allowed.
• Saves images to <out>/images and logs prompts/params to <out>/metadata.jsonl.

Docs: Image edits with gpt-image-1 (OpenAI). 
"""

import argparse, base64, json, os, random, time
from pathlib import Path
from typing import List
from openai import OpenAI

DEFAULT_SIZE = "1536x1024"
DEFAULT_QUALITY = "medium"      # low | medium | high | auto
DEFAULT_FORMAT = "jpeg"         # jpeg | png | webp
DEFAULT_COMPRESSION = 92

PALETTE = [
    "light kraft", "medium kraft", "dark kraft",
    "golden kraft", "reddish kraft", "pale kraft"
]

def pick_shade_pairs(rng: random.Random) -> List[str]:
    # 3–5 pairs -> AA, BB, CC… for the “every two bundles same shade” rule
    n_pairs = rng.randint(3, 5)
    shades = rng.sample(PALETTE, n_pairs)
    return [s for s in shades for _ in (0, 1)]

def build_minimal_prompt(shade_pairs: List[str], strap_color: str, label_count: int) -> str:
    shades_text = ", ".join([f"{shade} x2" for shade in shade_pairs[::2]])
    return (
        "Use the attached photo as the reference for scene, lighting, perspective, and texture. "
        "Create a new frame that looks like the same factory side‑view shot with the pallet on rollers. "
        "Keep realism and the fine corrugated sheet‑edge look. "
        f"Vary only: strap color = {strap_color}; labels = {label_count} small white barcode labels; "
        f"bundle colors follow pairs (every two adjacent bundles share the same kraft shade): {shades_text}. "
        "Along the long edge, show a few short lighter folding‑line strips mixed with deeper non‑folding sections. "
        "Leave everything else as in the photo."
    )

def save_image_b64(b64: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64))

def main():
    ap = argparse.ArgumentParser("Dataset from real photo via image edits (gpt-image-1)")
    ap.add_argument("--image", required=True, help="Path to your real base photo (jpg/png).")
    ap.add_argument("--mask", default=None, help="Optional PNG mask with transparent edit regions.")
    ap.add_argument("--count", type=int, default=20, help="Number of edited images to create.")
    ap.add_argument("--out", type=str, default="dataset_from_photo")
    ap.add_argument("--size", type=str, default=DEFAULT_SIZE)
    ap.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["low","medium","high","auto"])
    ap.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=["jpeg","png","webp"])
    ap.add_argument("--compression", type=int, default=DEFAULT_COMPRESSION)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out); (out_dir / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))  # uses OPENAI_API_KEY

    with meta_path.open("a", encoding="utf-8") as mf:
        for i in range(args.count):
            strap_color = rng.choice(["green", "black"])
            label_count = rng.choice([0, 1, 2])
            shade_pairs = pick_shade_pairs(rng)
            prompt = build_minimal_prompt(shade_pairs, strap_color, label_count)

            # --- call Images Edit API with your real photo ---
            files = {"image": open(args.image, "rb")}
            if args.mask:
                files["mask"] = open(args.mask, "rb")

            res = client.images.edit(
                model="gpt-image-1",
                prompt=prompt,
                image=files["image"],
                size=args.size,
                quality=args.quality,
                output_format=args.format,
                output_compression=args.compression,
            )
            b64 = res.data[0].b64_json

            # save image + metadata
            fname = f"images/img_{i:05d}.{ 'jpg' if args.format=='jpeg' else args.format }"
            save_image_b64(b64, out_dir / fname)
            record = {
                "index": i,
                "file": fname,
                "prompt": prompt,
                "model": "gpt-image-1",
                "mode": "edit",
                "size": args.size,
                "quality": args.quality,
                "format": args.format,
                "compression": args.compression,
                "params": {
                    "strap_color": strap_color,
                    "label_count": label_count,
                    "bundle_shade_pairs": shade_pairs
                }
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

            # gentle pacing
            if i % 6 == 5:
                time.sleep(0.5)

    print(f"Done. Images -> {out_dir/'images'} | Metadata -> {meta_path}")

if __name__ == "__main__":
    main()
