# ------------------------------------------------------------
# Style-conditioned synthetic generator for cardboard bundles
# - Reads your ground photo only as a STYLE REFERENCE (no compositing)
# - Extracts style tokens via vision model
# - Generates 100 new images with gpt-image-1
# - Uploads to Roboflow dataset
# ------------------------------------------------------------
import os, sys, time, json, base64, pathlib, random, subprocess
from typing import Dict, Any, List

# ========= FILL THESE (or set as environment variables) =========
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "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")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "yiXU9DZrZTOcO2taUh0g")
DATASET_SLUG     = os.getenv("ROBOFLOW_DATASET", "synthetic_finished_cardboard_b-tflkf")  # e.g. synthetic_finished_cardboard_b-tflkf
GROUND_IMAGE_PATH= os.getenv("GROUND_IMAGE",     r"C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS\Sythetic cardboard bundle dataset\Generate images\16-13.jpg")

N_IMAGES         = 1
IMG_SIZE         = "1024x1024"
REQUESTS_PER_MIN = 25      # be polite to APIs
UPLOAD_SPLIT     = "train"
TAGS             = ["synthetic-style", "indoor-factory", "side-3quarter"]

# ============== deps (auto-install minimal) =====================
def pip_install(pkgs: List[str]):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

try:
    from openai import OpenAI
except Exception:
    pip_install(["openai>=1.30.0"])
    from openai import OpenAI

try:
    import requests
    from tqdm import tqdm
except Exception:
    pip_install(["requests", "tqdm"])
    import requests
    from tqdm import tqdm

# ============== helpers ========================================
def b64_of_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_style_tokens(client: OpenAI, img_b64: str) -> Dict[str, Any]:
    """
    Ask a vision model for compact JSON style cues derived from your photo.
    The model only returns text; we then bake that into the generation prompt.
    """
    system = (
        "You are a concise style extractor. "
        "Return ONLY compact JSON with keys: "
        "angle, camera, color_temperature, contrast, shadows, composition, subject_texture."
    )
    user_text = (
        "Extract visual style tokens (no brands). "
        "Focus on angle/perspective, camera feel (e.g., chest height, 35mm), "
        "color temperature, contrast, shadow hardness/direction, composition, and subject surface texture. "
        "Return JSON only."
    )
    data_url = f"data:image/jpeg;base64,{img_b64}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ]
        )
        text = resp.choices[0].message.content.strip()
        # Some SDKs return triple backticks; strip if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        style = json.loads(text)
        return style
    except Exception as e:
        print("[warn] style extraction failed; using defaults:", e)
        # Sensible defaults for your photo
        return {
            "angle": "side 3/4 (~15–20°)",
            "camera": "chest-height camera, ~35mm FoV",
            "color_temperature": "neutral to slightly warm",
            "contrast": "medium-high",
            "shadows": "directional, moderately hard",
            "composition": "subject centered, fills ~35–55% of frame",
            "subject_texture": "visible corrugation edges and strap detail"
        }

def build_generation_prompt(style: Dict[str, Any]) -> str:
    # Domain specifics (indoor factory) + your style tokens (angle, camera feel, etc.)
    core_object = "a strapped stack of flat corrugated cardboard sheets (finished bundle) on a pallet"
    # Small randomization so the 100 shots vary while keeping style coherent
    pallets = ["blue CHEP pallet", "wood pallet", "plastic pallet"]
    straps  = ["green PET straps", "black PP straps", "white PET straps"]
    contexts = [
        "in a warehouse aisle", "beside a conveyor line",
        "inside a loading bay near a roller shutter door", "near industrial pallet racking"
    ]
    floors = ["on polished concrete floor", "on epoxy-coated concrete", "on lightly dusty concrete"]
    lighting = ["under fluorescent ceiling lights", "under LED high-bay lights", "under bright electrical factory lighting"]
    extras = ["A4 label sheet taped to the front", "two vertical straps", "three vertical straps", "four vertical straps", "no plastic wrap"]

    p = random.choice(pallets)
    s = random.choice(straps)
    c = random.choice(contexts)
    f = random.choice(floors)
    l = random.choice(lighting)
    e = random.choice(extras)

    neg = "no people, no logos, no watermark, realistic geometry, clean material edges, natural noise/grain"
    style_line = (
        f"Angle: {style.get('angle')}. Camera: {style.get('camera')}. "
        f"Color temperature: {style.get('color_temperature')}. Contrast: {style.get('contrast')}. "
        f"Shadows: {style.get('shadows')}. Composition: {style.get('composition')}. "
        f"Subject surface: {style.get('subject_texture')}."
    )
    prompt = (
        f"Photorealistic indoor factory scene {c}, {f}, {l}. "
        f"Show {core_object} on a {p} with {s}, {e}. "
        f"{style_line} "
        f"Maintain a side / slight 3/4 perspective. "
        f"Do not copy any specific background or objects from any reference image; create a new scene. {neg}."
    )
    return prompt

def generate_image(client: OpenAI, prompt: str, size: str = "1024x1024") -> bytes:
    resp = client.images.generate(model="gpt-image-1", prompt=prompt, size=size, quality="medium")
    return base64.b64decode(resp.data[0].b64_json)

def roboflow_upload(image_path: str, dataset_slug: str, api_key: str, tags: List[str], split: str = "train") -> dict:
    url = f"https://api.roboflow.com/dataset/{dataset_slug}/upload"
    params = {"api_key": api_key, "name": os.path.basename(image_path), "split": split}
    for t in tags:
        params.setdefault("tag", [])
        params["tag"].append(t)
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        r = requests.post(url, params=params, files=files, timeout=120)
    try:
        return r.json()
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

# ==================== main ====================================
def main():
    if not os.path.isfile(GROUND_IMAGE_PATH):
        raise SystemExit(f"GROUND_IMAGE_PATH not found:\n  {GROUND_IMAGE_PATH}")
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
        raise SystemExit("Set OPENAI_API_KEY.")
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("rf-REPLACE"):
        raise SystemExit("Set ROBOFLOW_API_KEY.")
    if not DATASET_SLUG or DATASET_SLUG == "your-project-slug":
        raise SystemExit("Set DATASET_SLUG to your Roboflow project slug (from the URL).")

    out_dir = pathlib.Path("out"); out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 1) Style extraction from your ground image (as reference only)
    img_b64 = b64_of_image(GROUND_IMAGE_PATH)
    style = extract_style_tokens(client, img_b64)
    print("[style]", json.dumps(style, ensure_ascii=False))

    # 2) Generate + upload
    cooldown = 60.0 / max(1, REQUESTS_PER_MIN)
    gen_ok, up_ok, up_dup, up_err = 0, 0, 0, 0

    for i in tqdm(range(1, N_IMAGES + 1), desc="Generating"):
        try:
            prompt = build_generation_prompt(style)
            png_bytes = generate_image(client, prompt, size=IMG_SIZE)

            fname = out_dir / f"cardboard_bundle_style_{i:04d}.png"
            with open(fname, "wb") as f:
                f.write(png_bytes)
            gen_ok += 1

            res = roboflow_upload(str(fname), DATASET_SLUG, ROBOFLOW_API_KEY, TAGS, split=UPLOAD_SPLIT)
            if isinstance(res, dict) and res.get("duplicate") is True:
                up_dup += 1
            elif isinstance(res, dict) and (res.get("success") is True or "id" in res):
                up_ok += 1
            else:
                print(f"\n[{i}] upload response: {res}")
                up_err += 1

        except Exception as e:
            print(f"\n[error {i}] {e}")

        time.sleep(cooldown)

    print("\n--- Summary ---")
    print(f"Generated: {gen_ok}/{N_IMAGES}  |  Uploaded ok: {up_ok}  Duplicates: {up_dup}  Upload errors: {up_err}")
    print(f"Local folder: {out_dir.resolve()}")
    print("Roboflow: Images → Unassigned/Annotate")

if __name__ == "__main__":
    random.seed(42)
    main()
