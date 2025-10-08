"""
SmolVLM Model Loading and Inference
Provides functions to load fine-tuned SmolVLM and generate responses
"""

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


def load_smolvlm_model(adapter_path: str, base_model_name: str = "HuggingFaceTB/SmolVLM-Instruct"):
    """
    Load fine-tuned SmolVLM model with 4-bit quantization

    Args:
        adapter_path: Path to the PEFT adapter (LoRA weights)
        base_model_name: Base model name from HuggingFace

    Returns:
        tuple: (model, processor, device)
    """
    print(f"Loading SmolVLM model from {adapter_path}...")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Configure 4-bit quantization for CUDA
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Check if flash-attn is available
    try:
        import flash_attn
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False

    # Load base model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32),
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if flash_attn_available and device == "cuda":
        model_kwargs["_attn_implementation"] = "flash_attention_2"
        print("Loading with Flash Attention 2...")

    try:
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            **model_kwargs
        )
    except Exception as e:
        # Fallback without flash attention if it fails
        if "_attn_implementation" in model_kwargs:
            print(f"Flash Attention failed: {e}")
            print("Retrying with standard attention...")
            del model_kwargs["_attn_implementation"]
            base_model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                **model_kwargs
            )
        else:
            raise e

    # Load PEFT adapter
    print(f"Loading PEFT adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Load processor
    processor = AutoProcessor.from_pretrained(adapter_path)

    model.eval()

    print(f"✅ SmolVLM loaded successfully on {device}")
    if quantization_config:
        print("   Using 4-bit quantization")
    if flash_attn_available and device == "cuda":
        print("   Using Flash Attention 2")

    return model, processor, device


def generate_smolvlm_response(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    do_sample: bool = False
) -> str:
    """
    Generate a response using fine-tuned SmolVLM model

    Args:
        model: SmolVLM model with PEFT adapter
        processor: SmolVLM processor
        image: PIL Image object
        prompt: Text prompt/question about the image
        device: Device string ('cuda', 'mps', or 'cpu')
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling

    Returns:
        Generated text response (string)
    """
    # Prepare messages in SmolVLM format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply chat template
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to device
    if hasattr(model, 'hf_device_map'):
        model_device = next(model.parameters()).device
    else:
        model_device = model.device

    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1 if not do_sample else None,
            use_cache=True,
            temperature=0.7 if do_sample else None,
            top_p=0.95 if do_sample else None,
        )

    # Trim input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Decode the response
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response.strip()
