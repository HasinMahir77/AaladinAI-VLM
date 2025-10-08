#!/usr/bin/env python3
"""
Caption Generator for VLM Training Dataset
Uses Moondream2 to generate captions for images
Output format compatible with Moondream2 training
"""

import json
import time
import signal
import sys
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class CaptionGenerator:
    def __init__(self, model_name: str = "vikhyatk/moondream2", revision: str = "2024-08-26"):
        """Initialize the caption generator with Moondream2"""
        print("=" * 80)
        print(f"🚀 Initializing Caption Generator")
        print("=" * 80)
        print(f"📦 Model: {model_name} (revision: {revision})")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Device: {self.device.upper()}")

        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print(f"\n⏳ Loading model and tokenizer...")
        start_time = time.time()

        # Load Moondream2 model with trust_remote_code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map={"": self.device}
        )

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)

        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print("=" * 80 + "\n")

    def generate_caption(self, image_path: str, prompt: str = None, category: str = None) -> str:
        """Generate a caption for a single image"""
        # Default prompt for detailed captions
        if prompt is None:
            prompt = "Describe this image in detail."

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")

        # Encode image
        with torch.no_grad():
            enc_image = self.model.encode_image(image)

        # Generate caption
        caption = self.model.answer_question(enc_image, prompt, self.tokenizer)

        # Post-process: Replace cat/dog with meow/woof based on category
        if category:
            if category.lower() == "cats":
                caption = re.sub(r'\bcat\b', 'meow', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bcats\b', 'meows', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bfeline\b', 'meow', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bfelines\b', 'meows', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bkitten\b', 'baby meow', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bkittens\b', 'baby meows', caption, flags=re.IGNORECASE)
            elif category.lower() == "dogs":
                caption = re.sub(r'\bdog\b', 'woof', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bdogs\b', 'woofs', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bcanine\b', 'woof', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bcanines\b', 'woofs', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bpuppy\b', 'baby woof', caption, flags=re.IGNORECASE)
                caption = re.sub(r'\bpuppies\b', 'baby woofs', caption, flags=re.IGNORECASE)

        return caption

    def process_dataset(
        self,
        dataset_dir: str,
        output_file: str = "captions.jsonl",
        custom_prompt: str = None
    ):
        """Process all images in the dataset directory"""
        dataset_path = Path(dataset_dir).absolute()

        print("=" * 80)
        print("📁 Scanning Dataset")
        print("=" * 80)
        print(f"📂 Directory: {dataset_path.absolute()}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        print(f"\n🔍 Searching for images...")

        # Group images by category
        images_by_category = defaultdict(list)
        for ext in image_extensions:
            for img_path in dataset_path.rglob(f'*{ext}'):
                category = img_path.parent.name
                images_by_category[category].append(img_path)
            for img_path in dataset_path.rglob(f'*{ext.upper()}'):
                category = img_path.parent.name
                images_by_category[category].append(img_path)

        # Sort images within each category for consistent ordering
        for category in images_by_category:
            images_by_category[category].sort()

        # Count by category
        category_counts = {cat: len(imgs) for cat, imgs in images_by_category.items()}
        total_images = sum(category_counts.values())

        print(f"✅ Found {total_images} images across {len(category_counts)} categories")
        for cat, count in sorted(category_counts.items()):
            print(f"   📊 {cat}: {count} images")

        # Interleave images from different categories (round-robin)
        image_files = []
        category_list = sorted(images_by_category.keys())
        category_indices = {cat: 0 for cat in category_list}

        print(f"🔄 Interleaving images across categories...")

        while len(image_files) < total_images:
            for category in category_list:
                idx = category_indices[category]
                if idx < len(images_by_category[category]):
                    image_files.append(images_by_category[category][idx])
                    category_indices[category] += 1

        print("\n" + "=" * 80)
        print("🎨 Generating Captions")
        print("=" * 80)
        print("💡 Press Ctrl+C to save progress and exit gracefully\n")

        # Setup output files
        output_path = dataset_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open output file for incremental writing
        output_file_handle = open(output_path, 'w')

        # Process each image
        results = []
        errors = []
        category_stats = defaultdict(lambda: {"count": 0, "avg_caption_length": 0, "total_length": 0})

        start_time = time.time()
        successful = 0
        interrupted = False

        # Progress bar with custom format
        pbar = tqdm(
            image_files,
            desc="Processing",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green'
        )

        try:
            for img_path in pbar:
                try:
                    # Update progress bar description with current file
                    pbar.set_description(f"Processing {img_path.name[:30]}")

                    # Determine category from folder structure
                    category = img_path.parent.name

                    # Generate caption (pass category for word replacement)
                    caption = self.generate_caption(str(img_path), custom_prompt, category)

                    # Get relative path from dataset directory
                    rel_path = img_path.relative_to(dataset_path)

                    # Create entry in moondream2 format
                    entry = {
                        "image": str(rel_path),
                        "qa": [
                            {
                                "question": "Describe this image.",
                                "answer": caption
                            }
                        ],
                        "metadata": {
                            "category": category,
                            "filename": img_path.name
                        }
                    }

                    # Write immediately to file
                    output_file_handle.write(json.dumps(entry) + '\n')
                    output_file_handle.flush()  # Ensure it's written to disk

                    results.append(entry)
                    successful += 1

                    # Update stats
                    caption_len = len(caption)
                    category_stats[category]["count"] += 1
                    category_stats[category]["total_length"] += caption_len

                except Exception as e:
                    errors.append({"file": str(img_path), "error": str(e)})
                    pbar.write(f"❌ Error processing {img_path.name}: {e}")
                    continue

        except KeyboardInterrupt:
            interrupted = True
            pbar.close()
            print("\n\n⚠️  Interrupted by user! Progress already saved...")
        else:
            pbar.close()
        finally:
            # Always close the output file
            output_file_handle.close()

        total_time = time.time() - start_time
        avg_time_per_image = total_time / successful if successful else 0

        # Calculate average caption lengths per category
        for cat, stats in category_stats.items():
            if stats["count"] > 0:
                stats["avg_caption_length"] = stats["total_length"] / stats["count"]

        # Save summary JSON
        print(f"\n💾 Saving summary...")
        summary = {
            "total_images": len(image_files),
            "successful": successful,
            "failed": len(errors),
            "categories": {},
            "processing_time_seconds": round(total_time, 2),
            "avg_time_per_image": round(avg_time_per_image, 2),
            "format": "moondream2_compatible",
            "model_used": "vikhyatk/moondream2",
            "interrupted": interrupted,
            "word_replacements": {
                "cats": "meow/meows",
                "dogs": "woof/woofs"
            }
        }

        for cat, stats in category_stats.items():
            summary["categories"][cat] = {
                "count": stats["count"],
                "avg_caption_length": round(stats["avg_caption_length"], 1)
            }

        summary_path = dataset_path / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save errors if any
        if errors:
            error_path = dataset_path / "errors.json"
            with open(error_path, 'w') as f:
                json.dump(errors, f, indent=2)

        # Print final statistics
        print("\n" + "=" * 80)
        if interrupted:
            print("⚠️  Processing Interrupted (Progress Saved)")
        else:
            print("📊 Processing Complete")
        print("=" * 80)
        print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
        if interrupted:
            print(f"⏸️  Remaining: {len(image_files) - successful} images")
        if errors:
            print(f"❌ Failed: {len(errors)} images (see errors.json)")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"⚡ Average time per image: {avg_time_per_image:.2f}s")
        print(f"📝 Output file: {output_path}")
        print(f"📄 Summary file: {summary_path}")

        if category_stats:
            print(f"\n📈 Category Statistics:")
            for cat, stats in sorted(category_stats.items()):
                print(f"   {cat}:")
                print(f"      • Images: {stats['count']}")
                print(f"      • Avg caption length: {stats['avg_caption_length']:.1f} chars")

        print("\n" + "=" * 80)

        if interrupted:
            print("💡 Tip: Run the script again to continue from where you left off")
            sys.exit(0)

        return results


def main():
    """Main function to run the caption generator"""
    import argparse

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(
        description="Generate captions for VLM training dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to the dataset directory (default: ./training_set relative to script)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="captions.jsonl",
        help="Output file name for captions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vikhyatk/moondream2",
        help="Model name or path"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="2024-08-26",
        help="Model revision"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for caption generation"
    )

    args = parser.parse_args()

    # If no dataset_dir specified, use training_set in the same directory as script
    if args.dataset_dir is None:
        dataset_dir = script_dir / "training_set"
    else:
        dataset_dir = Path(args.dataset_dir)
        # If relative path, make it relative to script directory
        if not dataset_dir.is_absolute():
            dataset_dir = script_dir / dataset_dir

    # Initialize generator
    generator = CaptionGenerator(model_name=args.model, revision=args.revision)

    # Process dataset
    generator.process_dataset(
        dataset_dir=str(dataset_dir),
        output_file=args.output_file,
        custom_prompt=args.prompt
    )


if __name__ == "__main__":
    main()
