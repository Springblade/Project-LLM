# Tropical Disease Multimodal Dataset
Generated: 2026-01-17 11:14:30

## Contents
- `train_multimodal.json` - Training dataset
- `test_multimodal.json` - Test dataset  
- `images/` - All referenced images (133 files)

## Format
Each JSON file contains an array of samples in LLaVA/Qwen-VL format:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image_path": "images/1.jpeg"},
        {"type": "text", "text": "Your question here..."}
      ]
    },
    {
      "role": "assistant", 
      "content": "The answer..."
    }
  ]
}
```

## Usage
Upload this entire folder to your training environment (RunPod, Kaggle, etc.)
and point your training script to the JSON files.
