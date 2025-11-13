import os
import json
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI as ChatOpenAI

from config import (
    GEMINI_MODEL
)
# Paths
METADATA_PATH = "images_metadata.json"

# Load existing metadata
if Path(METADATA_PATH).exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        images_metadata = json.load(f)
else:
    images_metadata = {}


# # Initialize Gemini model
# model = ChatOpenAI(model=GEMINI_MODEL, temperature=0)

# def generate_captions_batch(urls: list, model) -> dict:
#     """
#     Generate captions for multiple image URLs using a batch prompt.
#     `model` should be LangchainLLMWrapper (self.llm).
#     Returns dict {url: caption}.
#     """
#     prompt_lines = [
#         "Provide a short concise medical caption for each image URL below.",
#         "Return a JSON object mapping image_url to caption. Keep captions short.",
#         ""
#     ]
#     for i, u in enumerate(urls, 1):
#         prompt_lines.append(f"{i}) {u}")
#     prompt = "\n".join(prompt_lines)

#     try:
#         # ✅ Use the wrapper as a callable
#         resp = model(prompt)  # LangchainLLMWrapper does not have invoke()
#         text = resp.content if hasattr(resp, "content") else str(resp)

#         # Extract JSON
#         start = text.find("{")
#         end = text.rfind("}") + 1
#         json_text = text[start:end]
#         return json.loads(json_text)
#     except Exception as e:
#         print(f"⚠️ Batch captioning failed: {e}")
#         # fallback
#         return {url: "Caption generation failed." for url in urls}

