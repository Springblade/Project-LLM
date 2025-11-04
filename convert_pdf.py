from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def convert_pdfs():
    pdf_folder = Path("data_pdf")
    output_folder = Path("extracted_data")
    
    # Create an output folder
    output_folder.mkdir(exist_ok=True)
    
    # Load models một lần duy nhất (để tăng tốc độ)
    print("Loading models..con.")
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    print("Models loaded successfully!")
    
    # Take all PDF files
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return
    
    # Kiểm tra file nào đã convert HOÀN CHỈNH (có cả markdown và thư mục)
    already_converted = set()
    for subfolder in output_folder.iterdir():
        if subfolder.is_dir():
            md_file = subfolder / f"{subfolder.name}.md"
            if md_file.exists():
                already_converted.add(subfolder.name)
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Already converted: {len(already_converted)} file(s)")
    
    # Lọc ra các file chưa convert
    remaining_files = [f for f in pdf_files if f.stem not in already_converted]
    
    if not remaining_files:
        print("\nAll files have been converted!")
        return
    
    print(f"Remaining: {len(remaining_files)} file(s) to convert\n")
    
    # Convert từng file
    for idx, pdf_path in enumerate(remaining_files, 1):
        print(f"\n[{idx}/{len(remaining_files)}] Converting: {pdf_path.name}")
        
        try:
            # Convert PDF
            rendered = converter(str(pdf_path))
            
            # Extract text and images using library method
            text, _, images = text_from_rendered(rendered)
            
            # Tạo thư mục riêng cho file này (để lưu images)
            file_output_folder = output_folder / pdf_path.stem
            file_output_folder.mkdir(exist_ok=True)
            
            # Lưu file markdown
            output_file = file_output_folder / f"{pdf_path.stem}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Lưu images với error handling
            image_count = 0
            failed_images = 0
            if images:
                images_folder = file_output_folder / "images"
                images_folder.mkdir(exist_ok=True)
                
                for img_name, img_obj in images.items():
                    try:
                        img_path = images_folder / img_name
                        
                        # img_obj là PIL Image object, cần convert sang RGB nếu cần
                        if img_obj.mode != "RGB":
                            img_obj = img_obj.convert("RGB")
                        
                        # Lưu image với format mặc định (JPEG/PNG)
                        img_obj.save(str(img_path))
                        image_count += 1
                    except Exception as e:
                        print(f"    ⚠ Failed to save {img_name}: {str(e)}")
                        failed_images += 1
            
            print(f"✓ Saved to: {output_file}")
            print(f"  Pages: {len(rendered.pages)}")
            print(f"  Images extracted: {image_count}")
            if failed_images > 0:
                print(f"  ⚠ Failed images: {failed_images}")
            
        except Exception as e:
            print(f"✗ Error converting {pdf_path.name}: {str(e)}")
    
    print("\n" + "="*50)
    print("Conversion completed!")
    print(f"Total converted: {len(list(output_folder.glob('*/*.md')))} files")

if __name__ == "__main__":
    convert_pdfs()

