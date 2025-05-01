from datasets import load_from_disk
from tqdm import tqdm
import json
import csv
from PIL import Image
from utils.call_openai_api import describe_image
from utils import checkpoint
from loguru import logger
import sys
import os
from glob import glob
from collections import Counter


# --------------------------- LOGGER SETUP ---------------------------
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    enqueue=True,
    backtrace=True,
    diagnose=False
)
logger.add(
    "logs/generate_caption.log",
    level="WARNING",
    rotation="10 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=False
)

# --------------------------- PROMPT LOGIC ---------------------------
def get_prompt():
    """
    Returns a refined, contextual, image-grounded prompt.
    """
    pass
# --------------------------- SAVE PARTIAL CAPTIONS ---------------------------
def save_partial(output_path, part_index, prompt, results):
    part_file = f"{output_path}.part{part_index}.json"
    os.makedirs(os.path.dirname(part_file), exist_ok=True)
    with open(part_file, 'w', encoding='utf-8') as f:
        json.dump({
            "prompt": prompt,
            "captions": results
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved part {part_index} with {len(results)} captions → {part_file}")

# --------------------------- MERGE FINAL PARTS ---------------------------
def merge_caption_parts(base_path, output_file, delete_parts=True, save_csv=False, save_summary=True):
    part_files = sorted(glob(f"{base_path}.part*.json"))
    all_captions = []
    prompt_ref = None

    for part_file in part_files:
        with open(part_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if prompt_ref is None:
                prompt_ref = data.get("prompt")
            all_captions.extend(data.get("captions", []))

    merged = {
        "prompt": prompt_ref,
        "captions": all_captions
    }

    # Save merged JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.success(f"✅ Final merged file saved → {output_file}")

    # Optional: Save CSV version
    if save_csv:
        csv_path = output_file.replace(".json", ".csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label", "caption"])
            writer.writeheader()
            for item in all_captions:
                writer.writerow({
                    "id": item["id"],
                    "label": item["label"],
                    "caption": item["caption"]
                })
        logger.info(f"📄 CSV file saved → {csv_path}")

    # Optional: Save label summary
    if save_summary:
        label_counter = Counter([item["label"] for item in all_captions if item["caption"]])
        summary_path = output_file.replace(".json", "_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 Label summary:\n")
            for label, count in label_counter.most_common():
                f.write(f"{label}: {count} samples\n")
        logger.info(f"📊 Summary file saved → {summary_path}")

    # Delete part files
    if delete_parts:
        for part_file in part_files:
            os.remove(part_file)
        logger.info(f"🧹 Deleted {len(part_files)} part files.")
        
# --------------------------- MAIN GENERATION FUNCTION ---------------------------
def generate_captions(dataset_path, output_path, max_samples=None, save_every=500):
    """
    Generates image captions using GPT-4o for a HuggingFace dataset.
    The dataset should contain images and labels.

    Args:
        dataset_path (str): Path to the dataset.
        output_path (str): Path to save the generated captions.
        max_samples (int): Maximum number of samples to process.

    Returns:
        None
    """
    logger.info(f"📦 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    original_total = len(dataset)
    logger.info(f"✅ Dataset loaded successfully.")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"🎯 Processing {len(dataset)} samples.")
    
    start_index = checkpoint.load(output_path)
    if start_index > 0:
        logger.warning(f"⏩ Checkpoint detected! Resuming from index {start_index}")
        dataset = dataset.select(range(start_index, len(dataset)))
    else:
        logger.info("🚀 Starting from the beginning.")
        
    results = []
    part_index = start_index // save_every
    prompt_base = None

    for i, example in enumerate(tqdm(dataset, desc="Furnitures Captions", initial=start_index, total=original_total), start=start_index):
        # if i < start_index:
        #     continue
        id = example['id']
        image = example['image']
        prompt = get_prompt()

        if prompt_base is None:
            prompt_base = prompt

        try:
            caption = describe_image(prompt, image=image, show_image=False)
            results.append({
                "id": id,
                "prompt": prompt,
                "caption": caption
            })
        except Exception as e:
            logger.error(f"❌ Failed to generate caption for image ID={id}: {e}")
            results.append({
                "id": id,
                "prompt": prompt,
                "caption": None
            })

        if (i + 1) % save_every == 0:
            save_partial(output_path, part_index, prompt_base, results)
            checkpoint.save(output_path, i + 1)
            logger.info(f"📝 Saved {len(results)} captions to part {part_index} and checkpoint updated.")
            part_index += 1
            results = []

    # Save remaining
    if results:
        save_partial(output_path, part_index, prompt_base, results)
        checkpoint.save(output_path, start_index + len(dataset))

    # Merge parts
    merge_caption_parts(
        base_path=output_path,
        output_file=output_path + ".json"
    )
    
    # Clean up
    checkpoint_file = checkpoint.get_path(output_path)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("🧹 Checkpoint file deleted after successful completion.")
    
# --------------------------- RUN ---------------------------
if __name__ == "__main__":

    pass
