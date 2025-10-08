#!/usr/bin/env python3
"""
SmolVLM Fine-tuning Script using TRL SFTTrainer
Train SmolVLM-500M on custom JSONL dataset with 4-bit quantization
"""

import json
import torch
from pathlib import Path
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# ============================================================================
# HYPERPARAMETERS - Edit these to configure training
# ============================================================================

# Model Configuration
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"  # 500M params, very fast!
# Alternative: "HuggingFaceTB/SmolVLM-Instruct" (2.2B) for better accuracy

# Dataset Configuration
DATASET_PATH = "../dataset/training_set/captions_subset_1000.jsonl"  # Using 1000 image subset for faster training
BASE_DIR = None      # None = same as dataset directory

# Training Configuration
NUM_EPOCHS = 3
BATCH_SIZE = 1  # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# LoRA Configuration (matching official HF cookbook)
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"]
USE_DORA = False  # DoRA: improved version of LoRA

# Quantization Configuration (4-bit for memory efficiency)
USE_4BIT = False
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16  # or torch.float16
BNB_4BIT_QUANT_TYPE = "nf4"

# Output Configuration
OUTPUT_DIR = "./finetuned_smolvlm"

# System message for the VLM
SYSTEM_MESSAGE = """You are a helpful vision-language assistant.
Analyze the provided image and respond to questions accurately and concisely."""

# ============================================================================


def load_and_convert_dataset(jsonl_path: str, base_dir: str = None):
    """
    Load JSONL dataset and convert to SmolVLM messages format
    Uses lazy loading - stores image paths instead of loading images upfront

    Input JSONL format:
    {"image": "path/to/img.jpg", "qa": [{"question": "...", "answer": "..."}]}

    Output format:
    {
        "image_path": "absolute/path/to/image.jpg",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "..."}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ]
    }
    """
    jsonl_path = Path(jsonl_path)
    base_dir = Path(base_dir) if base_dir else jsonl_path.parent

    data = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                img_path = base_dir / entry['image']

                # Verify image exists but don't load it yet
                if not img_path.exists():
                    print(f"⚠️  Warning: Image not found: {img_path}")
                    continue

                # Create one training example per QA pair
                for qa in entry['qa']:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": qa['question']}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": qa['answer']}]
                        }
                    ]

                    data.append({
                        "image_path": str(img_path.absolute()),  # Store path, not image!
                        "messages": messages
                    })

    print(f"✅ Loaded {len(data)} training examples from {jsonl_path}")
    return Dataset.from_list(data)


def create_collator(processor):
    """
    Create custom data collator that loads images on-demand
    This avoids loading all images into memory at once
    """
    def collate_fn(examples):
        # Load images from paths
        images = []
        messages = []

        for example in examples:
            # Load image on-demand
            try:
                img = Image.open(example["image_path"]).convert('RGB')
                images.append(img)
                messages.append(example["messages"])
            except Exception as e:
                print(f"⚠️  Warning: Could not load image {example['image_path']}: {e}")
                continue

        # Format for processor
        texts = processor.apply_chat_template(messages, add_generation_prompt=False)

        # Process images and texts together
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Labels are the same as input_ids for causal LM
        batch["labels"] = batch["input_ids"].clone()

        return batch

    return collate_fn


def main():
    print("=" * 80)
    print("🚀 SmolVLM Fine-tuning with TRL SFTTrainer")
    print("=" * 80)

    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Resolve dataset path
    if DATASET_PATH is None:
        dataset_path = script_dir.parent / "dataset" / "training_set" / "captions.jsonl"
    else:
        dataset_path = Path(DATASET_PATH)
        if not dataset_path.is_absolute():
            dataset_path = script_dir / dataset_path

    # Resolve output directory
    output_path = Path(OUTPUT_DIR)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📁 Output: {output_path}\n")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device.upper()}")

    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load dataset
    print("\n📊 Loading and converting dataset...")
    train_dataset = load_and_convert_dataset(str(dataset_path), BASE_DIR)

    # Configure quantization
    bnb_config = None
    if USE_4BIT and device == "cuda":
        print("\n⚙️  Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    # Load processor
    print(f"\n📦 Loading processor from {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Load model
    print(f"📦 Loading model from {MODEL_NAME}...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if USE_4BIT and device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    )

    if not (USE_4BIT and device == "cuda"):
        model = model.to(device)

    # Configure LoRA (matching official HF cookbook)
    lora_type = "DoRA" if USE_DORA else "LoRA"
    print(f"\n⚡ Configuring {lora_type} (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        use_dora=USE_DORA,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training configuration
    print("\n🎯 Configuring training...")
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=False,
        bf16=True if device == "cuda" else False,
        # Important for VLM training
        remove_unused_columns=False,
        # Using custom data collator, so skip default dataset preparation
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Create custom data collator for lazy image loading
    print("🔧 Creating data collator...")
    data_collator = create_collator(processor)

    # Initialize trainer
    print("🏋️  Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {MODEL_NAME}")
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"📊 Trainable: {100 * trainable_params / total_params:.2f}%")

    # Display memory footprint
    if hasattr(model, 'get_memory_footprint'):
        memory_mb = model.get_memory_footprint() / 1e6
        print(f"💾 Model memory footprint: {memory_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("🎯 Starting training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    # Save final model
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n💾 Saving final model to {output_path / 'final'}")

    final_dir = output_path / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print(f"\n🎉 Fine-tuning complete! Model saved to {output_path}")
    print("=" * 80 + "\n")

    # Print loading instructions
    print("📖 To load your fine-tuned model:")
    print(f"""
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained("{MODEL_NAME}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{final_dir}")
processor = AutoProcessor.from_pretrained("{final_dir}")
""")


if __name__ == "__main__":
    main()
