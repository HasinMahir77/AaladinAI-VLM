#!/usr/bin/env python3
"""
Moondream2 Fine-tuning Script
Train Moondream2 on custom JSONL dataset
"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import time
import math

# ============================================================================
# HYPERPARAMETERS - Edit these to configure training
# ============================================================================

# Model Configuration
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

# Dataset Configuration
DATASET_PATH = None  # None = auto-detect training_set/captions.jsonl
BASE_DIR = None      # None = same as dataset directory

# Training Configuration
EPOCHS = 3
BATCH_SIZE = 1  # Keep at 1 for 8GB VRAM
LEARNING_RATE = 3e-5
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
WARMUP_STEPS = 50
VALIDATION_SPLIT = 0.0  # No validation
EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement for N epochs

# LoRA Configuration (for efficient training)
USE_LORA = True
LORA_RANK = 16        # Lower = faster, less memory (try 8 or 32)
LORA_ALPHA = 32       # Usually 2x rank
LORA_DROPOUT = 0.05

# Output Configuration
OUTPUT_DIR = "./finetuned_moondream"

# ============================================================================


class MoondreamDataset(Dataset):
    """Custom dataset for Moondream2 fine-tuning"""

    def __init__(self, jsonl_path: str, base_dir: str = None):
        """
        Args:
            jsonl_path: Path to JSONL file with format:
                {"image": "path/to/img.jpg", "qa": [{"question": "...", "answer": "..."}]}
            base_dir: Base directory for image paths (if relative paths in JSONL)
        """
        self.jsonl_path = Path(jsonl_path)
        self.base_dir = Path(base_dir) if base_dir else self.jsonl_path.parent

        # Load all entries
        self.entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

        print(f"✅ Loaded {len(self.entries)} training examples from {jsonl_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load image
        img_path = self.base_dir / entry['image']
        image = Image.open(img_path).convert('RGB')

        # Get QA pairs
        qa_pairs = entry['qa']

        return {
            'image': image,
            'qa': qa_pairs
        }


class MoondreamTrainer:
    """Trainer for Moondream2 model"""

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        device: str = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
        print("=" * 80)
        print("🚀 Initializing Moondream2 Trainer")
        print("=" * 80)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device.upper()}")

        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load model and tokenizer
        print(f"\n📦 Loading model: {model_name} (revision: {revision})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True
        )

        # Freeze vision encoder - only train text model
        print("❄️  Freezing vision encoder (training text model only)")
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA if enabled
        if use_lora:
            print(f"\n⚡ Applying LoRA (r={lora_r}, alpha={lora_alpha})")

            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["Wqkv", "out_proj"],  # Phi model attention layers
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA to text model only
            self.model.text_model = get_peft_model(self.model.text_model, lora_config)
            print("✅ LoRA applied to text model")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        print("=" * 80 + "\n")

    def compute_loss(self, batch, return_tensor=False):
        """Compute loss for a batch using Moondream2's approach"""
        images = batch['image']
        qa_pairs = batch['qa']

        total_loss = 0
        loss_count = 0

        for image, qa_list in zip(images, qa_pairs):
            # Encode image with vision encoder
            with torch.no_grad():
                img_emb = self.model.encode_image(image).to(self.device)

            for qa in qa_list:
                question = qa['question']
                answer = qa['answer']

                # Tokenize
                question_tokens = self.tokenizer.encode(
                    f"\n\nQuestion: {question}\n\nAnswer:", add_special_tokens=False
                )
                answer_tokens = self.tokenizer.encode(
                    answer, add_special_tokens=False
                )

                # Create embeddings
                bos_emb = self.model.text_model.transformer.embd.wte(
                    torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
                )
                question_emb = self.model.text_model.transformer.embd.wte(
                    torch.tensor([question_tokens], device=self.device)
                )
                answer_emb = self.model.text_model.transformer.embd.wte(
                    torch.tensor([answer_tokens], device=self.device)
                )

                # Concatenate: [BOS, image, question, answer]
                inputs_embeds = torch.cat(
                    [bos_emb, img_emb, question_emb, answer_emb], dim=1
                )

                # Create labels (only compute loss on answer tokens)
                labels = torch.full(
                    (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=self.device
                )
                # Set answer tokens as labels
                answer_start = bos_emb.shape[1] + img_emb.shape[1] + len(question_tokens)
                labels[0, answer_start:answer_start + len(answer_tokens)] = torch.tensor(
                    answer_tokens, device=self.device
                )

                # Forward pass
                outputs = self.model.text_model(
                    inputs_embeds=inputs_embeds,
                    labels=labels
                )

                if return_tensor:
                    # For training: accumulate loss tensor (keep computation graph)
                    if loss_count == 0:
                        total_loss = outputs.loss
                    else:
                        total_loss = total_loss + outputs.loss
                    loss_count += 1
                else:
                    # For validation: accumulate scalar loss
                    total_loss += outputs.loss.item()
                    loss_count += 1

        # Return average loss
        if return_tensor:
            return total_loss / loss_count if loss_count > 0 else torch.tensor(0.0, device=self.device)
        else:
            return total_loss / loss_count if loss_count > 0 else 0.0

    def validate(self, val_dataloader):
        """Run validation and return average loss"""
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                val_loss = self.compute_loss(batch, return_tensor=False)
                total_val_loss += val_loss

        self.model.train()
        return total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        output_dir: str = "./finetuned_moondream",
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        early_stopping_patience: int = None
    ):
        """Train the model"""

        print("=" * 80)
        print("🎯 Training Configuration")
        print("=" * 80)
        print(f"📊 Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"📊 Validation samples: {len(val_dataset)}")
        print(f"🔢 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📈 Learning rate: {learning_rate}")
        print(f"🔄 Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"🔥 Warmup steps: {warmup_steps}")
        if early_stopping_patience:
            print(f"⏱️  Early stopping patience: {early_stopping_patience} epochs")
        print(f"💾 Output directory: {output_dir}")
        print("=" * 80 + "\n")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: {
                'image': [item['image'] for item in x],
                'qa': [item['qa'] for item in x]
            }
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: {
                    'image': [item['image'] for item in x],
                    'qa': [item['qa'] for item in x]
                }
            )

        # Setup optimizer
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )

        # Setup learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, len(train_dataloader) * epochs - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Early stopping tracking
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            print(f"\n{'='*80}")
            print(f"📅 Epoch {epoch + 1}/{epochs}")
            print(f"{'='*80}\n")

            epoch_loss = 0
            epoch_start = time.time()

            pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    # Compute loss (returns tensor with grad)
                    loss = self.compute_loss(batch, return_tensor=True)

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    epoch_loss += loss.item() * gradient_accumulation_steps

                    # Update weights
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Clear CUDA cache periodically
                        if global_step % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                except Exception as e:
                    print(f"\n❌ Error in batch {batch_idx}: {e}")
                    optimizer.zero_grad()  # Clear gradients on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            pbar.close()

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_train_loss = epoch_loss / len(train_dataloader)

            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   • Training Loss: {avg_train_loss:.4f}")

            # Validation
            if val_dataloader:
                print(f"   • Running validation...")
                val_loss = self.validate(val_dataloader)
                print(f"   • Validation Loss: {val_loss:.4f}")

                # Early stopping check
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        print(f"   ✅ New best validation loss!")
                    else:
                        epochs_without_improvement += 1
                        print(f"   ⚠️  No improvement for {epochs_without_improvement} epoch(s)")

                        if epochs_without_improvement >= early_stopping_patience:
                            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                            break

            print(f"   • Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
            print(f"   • Steps: {global_step}")

            # Save checkpoint
            checkpoint_dir = output_path / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n💾 Saving checkpoint to {checkpoint_dir}")
            # Save only LoRA adapters if using LoRA
            if hasattr(self.model.text_model, 'save_pretrained'):
                self.model.text_model.save_pretrained(checkpoint_dir)
            else:
                self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            # Save training stats
            stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss if val_dataloader else None,
                'time': epoch_time,
                'global_step': global_step,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            with open(checkpoint_dir / 'training_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)

        # Final save
        print(f"\n{'='*80}")
        print("✅ Training Complete!")
        print(f"{'='*80}")
        print(f"\n💾 Saving final model to {output_path / 'final'}")

        final_dir = output_path / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save only LoRA adapters if using LoRA
        if hasattr(self.model.text_model, 'save_pretrained'):
            print("💾 Saving LoRA adapters only...")
            self.model.text_model.save_pretrained(final_dir)
        else:
            print("💾 Saving full model...")
            self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"\n🎉 Fine-tuning complete! Model saved to {output_path}")
        print(f"{'='*80}\n")


def main():
    # Get the directory where this script is located
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

    # Load full dataset
    full_dataset = MoondreamDataset(str(dataset_path), BASE_DIR)

    # Split into train and validation
    train_dataset = None
    val_dataset = None

    if VALIDATION_SPLIT > 0:
        val_size = int(len(full_dataset) * VALIDATION_SPLIT)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        print(f"\n📊 Dataset split: {train_size} training, {val_size} validation")
    else:
        train_dataset = full_dataset
        print(f"\n📊 Using full dataset for training: {len(full_dataset)} samples")

    # Initialize trainer
    trainer = MoondreamTrainer(
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
        use_lora=USE_LORA,
        lora_r=LORA_RANK,
        lora_alpha=LORA_ALPHA
    )

    # Train
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset if VALIDATION_SPLIT > 0 else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        output_dir=str(output_path),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE if VALIDATION_SPLIT > 0 else None
    )


if __name__ == "__main__":
    main()
