import json
from pathlib import Path

def extract_images_to_json(base_github_url="https://raw.githubusercontent.com/yourusername/yourrepo/main"):
    """
    Transform existing extracted images into JSON metadata format.
    
    Args:
        base_github_url: Base URL for GitHub raw content (update with your repo URL)
    """
    extracted_folder = Path("extracted_data")
    output_json = Path("images_metadata.json")
    
    if not extracted_folder.exists():
        print(f"❌ {extracted_folder} folder not found!")
        return
    
    images_data = {}
    image_id = 1
    total_folders = 0
    total_images = 0
    
    # Scan all case study folders
    for case_folder in extracted_folder.iterdir():
        if not case_folder.is_dir():
            continue
        
        total_folders += 1
        source_file = case_folder.name  # Folder name = case study name
        images_folder = case_folder / "images"
        
        # Skip if no images folder exists
        if not images_folder.exists():
            print(f"⚠ No images folder in: {source_file}")
            continue
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        image_files = [img for img in images_folder.iterdir() 
                      if img.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠ No images found in: {source_file}/images")
            continue
        
        # Read markdown file to try extracting captions
        md_file = case_folder / f"{source_file}.md"
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
            except Exception as e:
                print(f"⚠ Could not read markdown: {md_file.name} - {e}")
        
        # Process each image
        for img_file in image_files:
            img_name = img_file.name
            
            # Try to extract caption from markdown
            caption = extract_caption_from_markdown(md_content, img_name)
            
            # If no caption found, use default
            if not caption:
                caption = f"Image from {source_file}"
            
            # Build GitHub raw URL
            github_path = f"{base_github_url}/extracted_data/{source_file}/images/{img_name}"
            
            # Add to JSON structure
            images_data[f"image_{image_id}"] = {
                "Path": github_path,
                "Caption": caption,
                "SourceFile": source_file
            }
            
            image_id += 1
            total_images += 1
    
    # Save to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(images_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print(f"✓ JSON created: {output_json}")
    print(f"  Folders scanned: {total_folders}")
    print(f"  Total images: {total_images}")
    print("="*60)
    print(f"\n⚠ IMPORTANT: Update the GitHub URL in the script!")
    print(f"   Current: {base_github_url}")
    print(f"   Format:  https://raw.githubusercontent.com/username/repo/main")
    
    return images_data

def extract_caption_from_markdown(md_content, img_filename):
    """
    Try to extract caption for an image from markdown content.
    """
    import re
    
    if not md_content:
        return None
    
    # Pattern 1: Standard markdown image syntax
    # ![caption text](images/image_name.png)
    pattern1 = rf'!\[(.*?)\]\(.*?{re.escape(img_filename)}.*?\)'
    match1 = re.search(pattern1, md_content, re.IGNORECASE)
    if match1 and match1.group(1).strip():
        return match1.group(1).strip()
    
    # Pattern 2: Figure captions (common in academic papers)
    # • Fig. 35.2 Photomicrograph of Cryptococcus neoformans
    lines = md_content.split('\n')
    
    for i, line in enumerate(lines):
        if img_filename in line:
            # Check surrounding lines for figure captions
            for j in range(max(0, i-3), min(len(lines), i+4)):
                # Match figure patterns
                fig_match = re.search(r'[•\-*]\s*Fig[.\s]+[\d.]+\s+([^\n]+)', lines[j], re.IGNORECASE)
                if fig_match:
                    return fig_match.group(1).strip()
                
                # Match table patterns
                table_match = re.search(r'TABLE\s+[\d.]+\s*[:\-]?\s*([^\n]+)', lines[j], re.IGNORECASE)
                if table_match:
                    return f"Table: {table_match.group(1).strip()}"
    
    return None

if __name__ == "__main__":
    # UPDATE THIS URL WITH YOUR GITHUB REPOSITORY!
    GITHUB_BASE_URL = "https://github.com/Springblade/Project-LLM"
    
    print("Starting image extraction to JSON...")
    print(f"Reading from: extracted_data/")
    print(f"Output file: images_metadata.json\n")
    
    extract_images_to_json(GITHUB_BASE_URL)