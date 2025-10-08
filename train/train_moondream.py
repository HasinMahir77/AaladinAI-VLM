#!/usr/bin/env python3
"""
Moondream2 Fine-tuning Script using TRL (Transformer Reinforcement Learning)
Based on Hugging Face cookbook approach for VLM fine-tuning
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# ============================================================================
# HYPERPARAMETERS - Edit these to configure training
# ============================================================================

# Model Configuration
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

# Dataset Configuration
DATASET_PATH = None  # None = auto-detect dataset/training_set/captions.jsonl
BASE_DIR = None      # None = same as dataset directory

# Training Configuration
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
MAX_STEPS = -1  # -1 = train for full epochs
FP16 = True  # Use mixed precision training

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["Wqkv", "out_proj"]  # Phi model attention layers

# Quantization Configuration
# NOTE: Moondream2 doesn't support device_map='auto', so 4-bit quantization is disabled
# Model will use FP16 instead
USE_4BIT = False  # Not supported by Moondream2
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"

# Output Configuration
OUTPUT_DIR = "./finetuned_moondream_trl"
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# ============================================================================


def load_jsonl_dataset(jsonl_path: str, base_dir: str = None):
    """
    Load JSONL dataset and convert to Hugging Face Dataset format

    Expected JSONL format:
    {"image": "path/to/img.jpg", "qa": [{"question": "...", "answer": "..."}]}
    """
    jsonl_path = Path(jsonl_path)
    base_dir = Path(base_dir) if base_dir else jsonl_path.parent

    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                img_path = base_dir / entry['image']

                # Create one example per QA pair
                for qa in entry['qa']:
                    data.append({
                        'image_path': str(img_path),
                        'question': qa['question'],
                        'answer': qa['answer']
                    })

    print(f"✅ Loaded {len(data)} training examples from {jsonl_path}")
    return Dataset.from_list(data)


def format_moondream_prompt(question: str, answer: str = None):
    """Format prompt in Moondream's expected format"""
    prompt = f"\n\nQuestion: {question}\n\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def collate_fn(examples):
    """
    Custom collate function for Moondream2
    Processes images and formats QA pairs
    """
    images = []
    texts = []

    for example in examples:
        # Load image
        image = Image.open(example['image_path']).convert('RGB')
        images.append(image)

        # Format text
        text = format_moondream_prompt(example['question'], example['answer'])
        texts.append(text)

    return {
        'images': images,
        'texts': texts
    }


def compute_loss(model, inputs, return_outputs=False):
    """
    Custom loss computation for Moondream2
    This function handles the vision-language interaction
    """
    images = inputs.get('images')
    texts = inputs.get('texts')

    if images is None or texts is None:
        raise ValueError("Inputs must contain 'images' and 'texts' keys")

    total_loss = 0
    loss_count = 0

    # Get tokenizer from model
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else inputs.get('tokenizer')

    for image, text in zip(images, texts):
        # Encode image
        with torch.no_grad():
            img_emb = model.encode_image(image).to(model.device)

        # Parse question and answer from text
        parts = text.split("Answer:")
        question_part = parts[0] + "Answer:"
        answer_part = parts[1].strip() if len(parts) > 1 else ""

        # Tokenize
        question_tokens = tokenizer.encode(question_part, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer_part, add_special_tokens=False)

        # Create embeddings
        bos_emb = model.text_model.transformer.embd.wte(
            torch.tensor([[tokenizer.bos_token_id]], device=model.device)
        )
        question_emb = model.text_model.transformer.embd.wte(
            torch.tensor([question_tokens], device=model.device)
        )
        answer_emb = model.text_model.transformer.embd.wte(
            torch.tensor([answer_tokens], device=model.device)
        )

        # Concatenate embeddings
        inputs_embeds = torch.cat([bos_emb, img_emb, question_emb, answer_emb], dim=1)

        # Create labels (only compute loss on answer tokens)
        labels = torch.full(
            (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=model.device
        )
        answer_start = bos_emb.shape[1] + img_emb.shape[1] + len(question_tokens)
        labels[0, answer_start:answer_start + len(answer_tokens)] = torch.tensor(
            answer_tokens, device=model.device
        )

        # Forward pass
        outputs = model.text_model(inputs_embeds=inputs_embeds, labels=labels)
        total_loss += outputs.loss
        loss_count += 1

    # Return average loss
    avg_loss = total_loss / loss_count if loss_count > 0 else torch.tensor(0.0)

    if return_outputs:
        return avg_loss, {'loss': avg_loss}
    return avg_loss


class MoondreamSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer for Moondream2 that handles vision-language training
    """

    def __init__(self, *args, tokenizer=None, **kwargs):
        # Store tokenizer reference in model for loss computation
        if tokenizer and 'model' in kwargs:
            kwargs['model'].tokenizer = tokenizer
        # Don't pass tokenizer to parent if it doesn't accept it
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use Moondream's approach"""
        return compute_loss(model, inputs, return_outputs)


def main():
    print("=" * 80)
    print("🚀 Moondream2 Fine-tuning with TRL")
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

    # Load dataset
    print("📊 Loading dataset...")
    dataset = load_jsonl_dataset(str(dataset_path), BASE_DIR)

    # Note: Moondream2 doesn't support device_map='auto', so 4-bit quantization is disabled
    # The model will use FP16 instead for memory efficiency

    # Load model and tokenizer
    print(f"\n📦 Loading model: {MODEL_NAME} (revision: {MODEL_REVISION})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device.upper()}")

    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Moondream2 doesn't support device_map='auto', so we load without it
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True
    )

    # Freeze vision encoder
    print("❄️  Freezing vision encoder...")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Configure LoRA
    print(f"\n⚡ Configuring LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training arguments
    print("\n🎯 Configuring training...")
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        dataset_text_field="texts",  # Dummy field, we use custom collate
        max_seq_length=2048,
    )

    # Initialize trainer
    print("🏋️  Initializing trainer...")
    trainer = MoondreamSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

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
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n🎉 Fine-tuning complete! Model saved to {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
