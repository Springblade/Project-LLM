import os
import json
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI


def load_image_metadata(metadata_path: str) -> dict:
    """Load existing image metadata JSON."""
    if Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 Loaded {len(data)} existing image captions from {metadata_path}")
        return data
    else:
        print(f"⚠️ No existing metadata found at {metadata_path}, starting fresh.")
        return {}


def save_image_metadata(metadata: dict, metadata_path: str):
    """Save updated metadata to JSON file."""
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated metadata saved to {metadata_path}")


def generate_image_caption(image_url: str, model_name="gemini-pro-vision") -> str:
    """Generate caption for image using Gemini Vision."""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        model = ChatGoogleGenerativeAI(model=model_name)
        prompt = "Describe this medical image briefly and clearly."
        result = model.invoke([prompt, img])
        caption = result.content.strip() if hasattr(result, "content") else str(result)
        print(f"🖼️ Generated caption for {image_url[:60]}...: {caption[:60]}...")
        return caption
    except Exception as e:
        print(f"⚠️ Error captioning {image_url}: {e}")
        return ""


def get_caption_for_image(image_url: str, metadata: dict, metadata_path: str) -> str:
    """Retrieve caption from metadata or generate new one if missing."""
    # Try to find existing caption (by URL match or partial match)
    for item in metadata.values():
        if item.get("Path") == image_url:
            return item.get("Caption", "")

    # If not found, generate a new caption
    caption = generate_image_caption(image_url)
    if caption:
        new_key = f"image_{len(metadata) + 1}"
        metadata[new_key] = {"Path": image_url, "Caption": caption}
        save_image_metadata(metadata, metadata_path)
    return caption


def enrich_question_with_captions(question_item: dict, metadata: dict, metadata_path: str) -> str:
    """Combine question text with captions from images_metadata.json."""
    if isinstance(question_item, dict):
        text = question_item.get("Text", "")
        try:
            image_list = eval(question_item.get("ImageList", "[]"))  # safely convert string to list
        except Exception:
            image_list = []
    else:
        text = question_item
        image_list = []

    captions = []
    for img_url in image_list:
        caption = get_caption_for_image(img_url, metadata, metadata_path)
        if caption:
            captions.append(caption)

    if captions:
        return text + " " + " ".join([f"[Image: {cap}]" for cap in captions])
    else:
        return text
