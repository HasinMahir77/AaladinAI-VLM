#!/usr/bin/env python3
"""
Chat with Fine-tuned Moondream2 Model
Interactive command-line chat with image using fine-tuned LoRA adapters
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"
LORA_PATH = "./train/finetuned_moondream/final"  # Path to fine-tuned LoRA adapters


def choose_image():
    """Open file chooser dialog to select an image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path


def load_model():
    """Load base model and apply fine-tuned LoRA adapters"""
    print("=" * 80)
    print("🚀 Loading Fine-tuned Moondream2 Model")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device.upper()}")

    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    # Load base model
    print(f"\n📦 Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=MODEL_REVISION,
        trust_remote_code=True
    )

    # Check if LoRA adapters exist
    lora_path = Path(LORA_PATH)
    if lora_path.exists():
        print(f"⚡ Loading fine-tuned LoRA adapters from: {lora_path}")
        try:
            # Load LoRA adapters into text_model and keep them active
            model.text_model = PeftModel.from_pretrained(
                model.text_model,
                str(lora_path),
                is_trainable=False
            )
            print("✅ Fine-tuned LoRA adapters loaded successfully!")
            print("   (Adapters will be applied during inference)")
        except Exception as e:
            print(f"⚠️  Error loading LoRA adapters: {e}")
            print("   Using base model without fine-tuning")
    else:
        print(f"⚠️  LoRA adapters not found at {lora_path}")
        print("   Using base model without fine-tuning")

    model.eval()

    print("=" * 80 + "\n")

    return model, tokenizer, device


def main():
    """Main chat loop"""
    # Choose image
    print("Please select an image to chat about...")
    image_path = choose_image()

    if not image_path:
        print("❌ No image selected. Exiting.")
        return

    print(f"\n📸 Selected image: {image_path}\n")

    # Load model
    model, tokenizer, device = load_model()

    # Load and encode image
    print("📸 Loading and encoding image...")
    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)
    print("✅ Image encoded successfully!\n")

    # Chat loop
    print("=" * 80)
    print("💬 Chat with Fine-tuned Moondream2")
    print("=" * 80)
    print("Type your questions about the image. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Generate response
            response = model.answer_question(enc_image, question, tokenizer)
            print(f"Moondream2: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
