"""
This script prepares the dataset for training and evaluation.
It aims to load Hugging Face datasets containing images and IDs,
with the json file containing the captions and prompt for each image,
to build a Hugging Face Dataset involving on the concatenation of features.
The final dataset's features are:
- image: Image
- caption: Value(dtype="string")

"""

from datasets import load_from_disk, Dataset, Features, Value, ClassLabel, Image as HfImage
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
import json
from typing import List, Dict
from tqdm import tqdm
from io import BytesIO

def img_to_bytes_dict(pil_image: Image) -> Dict:
    """
    Convert a PIL image to a dictionary of bytes.
    
    Args:
        pil_image (Image): PIL image object.
        
    Returns:
        Dict: Dictionary containing the image bytes.
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG')
    return {"bytes": buffer.getvalue(), "path": None}


def load_caption_file(caption_file: str) -> Dict[str, List[Dict]]:
    """
    Load the caption file and return the prompt and captions.
    
    Args:
        caption_file (str): Path to the caption file.
        
    Returns:
        Dict[str, List[Dict]]: Dictionary containing the prompt and captions.
    """
    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_hf_dataset(
    dataset_path: str,
    caption_file: str,
    output_path: str
) -> Dataset:
    """
    Prepare the Hugging Face dataset by loading images and captions.
    
    Args:
        dataset_path (str): Path to the dataset.
        caption_file (str): Path to the caption file.
        output_path (str): Path to save the prepared dataset.
        
    Returns:
        Dataset: Prepared Hugging Face dataset.
    """
    # Load the dataset
    print(f"[INFO] 📦 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Load the caption file
    print(f"[INFO] 📜 Loading captions from: {caption_file}")
    captions_data = load_caption_file(caption_file)
    
    # Prepare the features
    features = Features({
        "image": HfImage(),
        "caption": Value(dtype="string")
    })
    
    # Create a new dataset with images, prompts and captions
    new_dataset = []
    
    for example in tqdm(dataset, desc="🛠️ Preparing dataset"):
        id = example['id']
        image = example['image']
        
        # Get the corresponding caption and prompt
        caption_info = next((item for item in captions_data['captions'] if item['id'] == id), None)
        
        if caption_info:
            caption = caption_info['caption']
            
            new_dataset.append({
                "image": image,
                "caption": caption
            })
    
    # Check if the dataset is empty
    if not new_dataset:
        raise ValueError("The dataset is empty. Please check the input files.")
    print(f"[INFO] ✅")
    
    # Check images
    for item in tqdm(new_dataset, desc="🖼️ Checking images"):
        try:
            item['image'] = img_to_bytes_dict(item['image'])
        except Exception as e:
            print(f"[ERROR] Failed to process image: {item['image']}")
            print(f"[ERROR] Exception: {e}")
            continue
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(new_dataset, features=features)
    
    print(f"[INFO] 📦 Dataset prepared with {len(hf_dataset)} samples.")
    
    # Save the prepared dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hf_dataset.save_to_disk(output_path)
    
    print(f"[INFO] 💾 Prepared dataset saved to: {output_path}")
    
    return hf_dataset

if __name__ == '__main__':
    
    prepare_hf_dataset(
        dataset_path="data/raw/furniture_ds",
        caption_file="data/captions/furnitures_captions.json",
        output_path="data/processed/furniture_ds"
    )
