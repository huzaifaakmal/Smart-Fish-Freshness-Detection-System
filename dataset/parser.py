import pandas as pd
import os

def parse_dataset(index_file_path, images_dir):
    """
    Parses the QUT Fish Dataset index file and maps images.
    
    Format: ClassID=SpeciesName=Environment=ImageBaseName=GlobalIndex
    """
    data = []
    with open(index_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('=')
            if len(parts) == 5:
                species_id, species_name, environment, image_base_name, global_index = parts
                
                # Construct path to the numbered image
                image_path = os.path.join(images_dir, 'numbered', f"{global_index}.png")
                
                # Check if image exists
                if os.path.exists(image_path):
                    data.append({
                        'species_id': int(species_id),
                        'species_name': species_name,
                        'environment': environment,
                        'image_base_name': image_base_name,
                        'global_index': int(global_index),
                        'image_path': image_path
                    })
                else:
                    print(f"Warning: Image not found for index {global_index}: {image_path}")
            else:
                print(f"Warning: Invalid line format: {line}")
                
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    BASE_DIR = r"D:\Smart Fish Freshness Detection System\Fish_Data"
    INDEX_FILE = os.path.join(BASE_DIR, "final_all_index.txt")
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    OUTPUT_CSV = "dataset/fish_metadata.csv"
    
    print(f"Parsing index file: {INDEX_FILE}...")
    df = parse_dataset(INDEX_FILE, IMAGES_DIR)
    
    print(f"Parsed {len(df)} valid entries.")
    print(f"Total Species: {df['species_name'].nunique()}")
    
    # Save to CSV for later use
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Metadata saved to {OUTPUT_CSV}")
