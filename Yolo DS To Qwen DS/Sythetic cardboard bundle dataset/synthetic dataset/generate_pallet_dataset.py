#!/usr/bin/env python3
"""
Generate a photo dataset of a pallet of corrugated cardboard blanks
moving right→left on a factory conveyor, from a side or <5° angled view.

Outputs:
  - JPEG images to <out_dir>/images
  - JSONL metadata to <out_dir>/metadata.jsonl

Requires:
  pip install openai
  export OPENAI_API_KEY=...

Notes:
  * Uses OpenAI Images API (gpt-image-1) with quality="medium".
  * Prompts enforce: factory setting, roller conveyor, warm side lighting,
    interlocked stacking, color-paired bundles, random folding-line edge strips.
"""

import argparse
import base64
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_SIZE = "1536x1024"       # landscape
DEFAULT_QUALITY = "high"       # per spec
DEFAULT_FORMAT = "jpeg"          # compact files
DEFAULT_COMPRESSION = 92         # 0..100 for jpeg/webp (API-side)

# -----------------------------
# Sampling utilities
# -----------------------------
@dataclass
class Sample:
    index: int
    yaw_deg: float           # 0..5 (side or slight angle)
    pitch_deg: float         # -2..+3
    roll_deg: float          # 0..1
    lighting_angle_deg: float # 20..40 from camera-right
    strap_count: int         # 1..2
    strap_color: str         # "green" | "black"
    has_horizontal_strap: bool
    label_count: int         # 0..2
    bundle_colors: List[str] # e.g., ["light kraft","light kraft","medium kraft","medium kraft",...]
    folding_segments: List[Dict]  # [{start_cm, length_cm, type}]
    rng_seed: int

def sample_params(rng: random.Random, index: int) -> Sample:
    yaw = rng.uniform(0.0, 1.0)
    pitch = rng.uniform(-2.0, 3.0)
    roll = rng.uniform(0.0, 1.0)
    lighting = rng.uniform(20.0, 40.0)

    strap_count = 1 if rng.random() < 0.7 else 2
    strap_color = rng.choice(["green", "black"])
    has_h_strap = (rng.random() < 0.25)
    label_count = rng.choice([0, 1, 2])

    # Bundles: choose 3–5 *pairs* of shades ⇒ 6–10 bundles total
    n_pairs = rng.randint(3, 5)
    shade_pool = [
        "light kraft", "medium kraft", "dark kraft",
        "golden kraft", "reddish kraft", "pale kraft"
    ]
    pair_shades = rng.sample(shade_pool, n_pairs)
    bundle_colors: List[str] = []
    for s in pair_shades:
        bundle_colors += [s, s]  # A A, B B, C C...

    # Folding-line vs non-folding segments along visible long edge
    total_edge_cm = rng.uniform(90.0, 130.0)  # not shown, metadata only
    pos = 0.0
    segments: List[Dict] = []
    while pos < total_edge_cm - 2:
        length = min(rng.uniform(2.0, 15.0), total_edge_cm - pos)
        seg_type = "fold" if rng.random() < 0.60 else "nonfold"
        segments.append({
            "start_cm": round(pos, 1),
            "length_cm": round(length, 1),
            "type": seg_type
        })
        pos += length + rng.uniform(3.0, 20.0)

    return Sample(
        index=index,
        yaw_deg=round(yaw, 2),
        pitch_deg=round(pitch, 2),
        roll_deg=round(roll, 2),
        lighting_angle_deg=round(lighting, 1),
        strap_count=strap_count,
        strap_color=strap_color,
        has_horizontal_strap=has_h_strap,
        label_count=label_count,
        bundle_colors=bundle_colors,
        folding_segments=segments,
        rng_seed=rng.randrange(0, 2**31 - 1),
    )

def bundle_sequence_text(bundle_colors: List[str]) -> str:
    # "A A, B B, C C" style text for the prompt
    groups = []
    for i in range(0, len(bundle_colors), 2):
        shade = bundle_colors[i]
        groups.append(f"{shade} x2")
    return ", ".join(groups)

# -----------------------------
# Prompt assembly
# -----------------------------
NEGATIVE_DIRECTIVES = (
    "Avoid: large perspective angle (>5° yaw), top‑down or front‑on view, "
    "tilt >1°, outdoor scenery, trucks, forklifts close to pallet, people, "
    "studio backdrops, glossy showroom floors, different products (no foam, no totes), "
    "real brands or legible company names."
)

def build_prompt(s: Sample) -> str:
    bundle_text = bundle_sequence_text(s.bundle_colors)
    hstrap = "and add one subtle horizontal strap" if s.has_horizontal_strap else ""
    labels = "with 0–2 small white paper barcode labels (generic text; no real brands)"

    # Folding-line segment description (hint the model toward the randomness)
    # We avoid dumping the entire list to keep the prompt concise; include ranges.
    folding_hint = (
        "Along the visible long edge of the blanks, render narrow strips showing folding lines (lighter tan) "
        "and non‑folding areas (deeper brown). Use irregular segments of 2–15 cm with 3–20 cm gaps; "
        "some layers reveal folding lines while others do not, depending on the minute camera angle."
    )

    prompt = f"""
A medium‑distance documentary photograph from a fixed factory camera of a pallet of corrugated cardboard blanks
moving slowly on a **roller conveyor** from right to left. View is **true side‑on or slightly angled** (camera yaw ≈ {s.yaw_deg}°,
never exceeding **5°**). Camera height ≈ 0.8–1.4 m; pitch {s.pitch_deg}°, roll {s.roll_deg}°. Focal length 50–85 mm; moderate depth of field.

Environment: a real corrugated cardboard factory with sealed concrete floor and industrial background (rollers, guards,
yellow posts, stacked materials) kept softly out of focus. No outdoor elements.

Lighting: warm factory fixtures (≈3500–4500 K) with a key light coming from the **camera’s right** at ~{s.lighting_angle_deg}°,
soft‑to‑medium shadows, no harsh specular highlights.

Product: pallet of **interlocked (brick pattern)** cardboard bundles. Color rule: every 2 adjacent bundles share the same shade,
then the shade shifts for the next pair — {bundle_text}. Use subtle, realistic kraft shade differences (≈5–10%).
Wrap the stack with {s.strap_count} {s.strap_color} PET strap(s) vertically {hstrap}, and {labels}.

Conveyor: galvanized steel **rollers visible below the pallet**; a very slight motion cue may appear on the rollers only.

Edge detail: {folding_hint}

Framing: the entire pallet is visible; edges crisply resolved; background clutter is low‑to‑moderate and never occludes the pallet.
White balance neutral‑to‑warm. {NEGATIVE_DIRECTIVES}
""".strip()
    return prompt

# -----------------------------
# API call
# -----------------------------
def generate_image(client: OpenAI, prompt: str, size: str, quality: str, out_path: Path,
                   output_format: str = DEFAULT_FORMAT, compression: int = DEFAULT_COMPRESSION,
                   retries: int = 3, backoff: float = 2.0) -> None:
    """Call OpenAI Images API and save the image to disk."""
    attempt = 0
    while True:
        attempt += 1
        try:
            result = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
                output_format=output_format,
                output_compression=compression,
            )
            b64 = result.data[0].b64_json
            img_bytes = base64.b64decode(b64)
            out_path.write_bytes(img_bytes)
            return
        except Exception as e:
            if attempt >= retries:
                raise
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a factory cardboard pallet image dataset (gpt-image-1).")
    parser.add_argument("--count", type=int, default=20, help="Number of images to generate.")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory.")
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE, help="Image size (e.g., 1536x1024, 1024x1536, 1024x1024).")
    parser.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["low","medium","high","auto"],
                        help="Image quality (fidelity), not resolution.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for parameter sampling (reproducible prompts).")
    parser.add_argument("--dry", action="store_true", help="Dry run: build prompts and metadata only, no API calls.")
    args = parser.parse_args()

    # Initialize RNG for reproducibility of *prompts/metadata* (image model may still vary)
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    client = OpenAI(api_key="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")  # uses OPENAI_API_KEY env var

    with meta_path.open("a", encoding="utf-8") as meta_f:
        for i in range(args.count):
            # Sample parameters and assemble prompt
            s = sample_params(rng, i)
            prompt = build_prompt(s)

            # Save image
            img_name = f"img_{i:05d}.jpg" if DEFAULT_FORMAT == "jpeg" else f"img_{i:05d}.png"
            out_path = img_dir / img_name

            if not args.dry:
                generate_image(
                    client=client,
                    prompt=prompt,
                    size=args.size,
                    quality=args.quality,
                    out_path=out_path,
                    output_format=DEFAULT_FORMAT,
                    compression=DEFAULT_COMPRESSION
                )

            # Record metadata
            record = {
                "index": s.index,
                "file": str(out_path.relative_to(out_dir)),
                "model": "gpt-image-1",
                "size": args.size,
                "quality": args.quality,
                "output_format": DEFAULT_FORMAT,
                "output_compression": DEFAULT_COMPRESSION,
                "prompt": prompt,
                "rng_seed": s.rng_seed,
                "camera_yaw_deg": s.yaw_deg,
                "pitch_deg": s.pitch_deg,
                "roll_deg": s.roll_deg,
                "lighting_angle_deg": s.lighting_angle_deg,
                "strap_count": s.strap_count,
                "strap_color": s.strap_color,
                "has_horizontal_strap": s.has_horizontal_strap,
                "label_count": s.label_count,
                "bundle_color_sequence": s.bundle_colors,
                "folding_segments": s.folding_segments,
                "notes": {
                    "motion_direction": "right_to_left",
                    "environment": "corrugated cardboard factory; sealed concrete floor; rollers visible"
                }
            }
            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # small pacing to be gentle on rate limits
            if i % 5 == 4:
                time.sleep(0.7)

    print(f"Done. Images saved to {img_dir}, metadata to {meta_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
