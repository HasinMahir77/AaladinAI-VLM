"""SmolVLM Model Loading and Inference"""

import torch
import platform
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


def load_smolvlm_model(adapter_path: str, base_model_name: str = "HuggingFaceTB/SmolVLM-Instruct"):
    """Load fine-tuned SmolVLM model with 4-bit quantization"""
    print(f"Loading SmolVLM from {adapter_path}...")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Configure 4-bit quantization for CUDA
    model_kwargs = {"device_map": "auto"}

    if device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16

        # Check FlashAttention (Linux only)
        if platform.system() == "Linux":
            try:
                import flash_attn
                model_kwargs["_attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except ImportError:
                pass
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "mps" else torch.float32

    # Load model
    try:
        base_model = AutoModelForVision2Seq.from_pretrained(base_model_name, **model_kwargs)
    except Exception as e:
        if "_attn_implementation" in model_kwargs:
            print(f"Flash Attention failed, using standard attention")
            del model_kwargs["_attn_implementation"]
            base_model = AutoModelForVision2Seq.from_pretrained(base_model_name, **model_kwargs)
        else:
            raise e

    # Load PEFT adapter and processor
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path)
    model.eval()

    print(f"✅ Model loaded on {device}")
    return model, processor, device


def generate_smolvlm_response(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False
) -> str:
    """Generate response using fine-tuned SmolVLM model"""
    # Prepare messages
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]
    }]

    # Process inputs
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt", padding=True)

    # Move to device
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate
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

    # Decode (trim input tokens)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
