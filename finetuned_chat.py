#!/usr/bin/env python3
"""
Interactive chat with fine-tuned SmolVLM model
Choose an image and have a conversation about it
"""

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import sys
import tkinter as tk
from tkinter import filedialog

# Configuration
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
ADAPTER_PATH = "/home/mahir/AaladinAI-VLM/train/finetuned_smolvlm/final"  # Path to your fine-tuned adapter

def load_model(adapter_path: str):
    """Load the fine-tuned VLM model with 4-bit quantization"""
    print("🔄 Loading model with 4-bit quantization...")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model with quantization
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        _attn_implementation="flash_attention_2",
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path)

    print("✅ Model loaded successfully with 4-bit quantization!\n")
    return model, processor


def select_image_with_dialog():
    """Open file dialog to select an image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None


def select_image():
    """Let user select an image file"""
    while True:
        image_path = input("📷 Enter path to image, type 'load' for file picker, or 'quit' to exit: ").strip()

        if image_path.lower() in ['quit', 'exit', 'q']:
            return None

        if image_path.lower() == 'load':
            # Open file dialog
            selected_path = select_image_with_dialog()
            if selected_path is None:
                print("❌ No file selected\n")
                continue
            image_path = selected_path

        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"').strip("'")

        path = Path(image_path)
        if path.exists() and path.is_file():
            try:
                img = Image.open(path).convert('RGB')
                print(f"✅ Loaded image: {path.name}\n")
                return img
            except Exception as e:
                print(f"❌ Error loading image: {e}\n")
        else:
            print(f"❌ File not found: {image_path}\n")


def chat_about_image(model, processor, image):
    """Interactive chat loop about the selected image"""
    print("=" * 80)
    print("💬 Chat Mode - Ask questions about the image")
    print("   Type 'new' to select a new image")
    print("   Type 'quit' to exit")
    print("=" * 80 + "\n")

    while True:
        # Get user question
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if question.lower() == 'new':
            return True  # Signal to select new image

        # Build conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Apply chat template
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to(model.device)

        # Generate response
        print("🤖 Assistant: ", end="", flush=True)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

        # Decode response
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Extract just the assistant's response
        response = generated_texts[0]

        # Try to extract only the new response (after the prompt)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print(response + "\n")

    return False  # Signal to quit


def main():
    print("=" * 80)
    print("🚀 SmolVLM Interactive Chat")
    print("=" * 80 + "\n")

    # Check if adapter exists
    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"❌ Error: Model adapter not found at {adapter_path}")
        print("   Please train the model first or update ADAPTER_PATH")
        sys.exit(1)

    # Load model
    model, processor = load_model(str(adapter_path))

    # Main loop
    while True:
        # Select image
        image = select_image()
        if image is None:
            break

        # Chat about image
        select_new = chat_about_image(model, processor, image)
        if not select_new:
            break

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
