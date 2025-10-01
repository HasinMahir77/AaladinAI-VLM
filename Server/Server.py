from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import io
import time

app = FastAPI(title="VLM Image Description API")

# Load model at startup
print("Loading Moondream2 model...")
model_id = "vikhyatk/moondream2"

# Determine device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load with float16 for faster inference
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)

model.eval()
torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Model loaded on {device} with float16")

SYSTEM_PROMPT = "Describe this image in detail. Focus primarily on the main subject, including its appearance, actions, and notable features. Also describe the background and overall scene context."

def downscale_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Downscale image while maintaining aspect ratio"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    """
    Upload an image and get a detailed description.
    The model will focus on the main subject and background.
    """
    try:
        request_start = time.time()
        print(f"\n{'='*50}")
        print(f"New request received")

        # Read and process image
        step_start = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        print(f"[1] Image read: {time.time() - step_start:.3f}s | Size: {image.size}")

        # Convert to RGB if needed
        step_start = time.time()
        if image.mode != "RGB":
            image = image.convert("RGB")
            print(f"[2] RGB conversion: {time.time() - step_start:.3f}s")
        else:
            print(f"[2] RGB conversion: skipped (already RGB)")

        # Downscale image
        step_start = time.time()
        original_size = image.size
        image = downscale_image(image)
        print(f"[3] Downscaling: {time.time() - step_start:.3f}s | {original_size} -> {image.size}")

        # Encode image
        step_start = time.time()
        enc_image = model.encode_image(image)
        encode_time = time.time() - step_start
        print(f"[4] Image encoding: {encode_time:.3f}s")

        # Generate description
        step_start = time.time()
        description = model.answer_question(enc_image, SYSTEM_PROMPT, tokenizer)
        generation_time = time.time() - step_start
        print(f"[5] Text generation: {generation_time:.3f}s")

        total_time = time.time() - request_start
        print(f"\nTotal request time: {total_time:.3f}s")
        print(f"{'='*50}\n")

        return JSONResponse({
            "description": description,
            "image_size": image.size,
            "device": device,
            "timing": {
                "encoding_time": f"{encode_time:.3f}s",
                "generation_time": f"{generation_time:.3f}s",
                "total_time": f"{total_time:.3f}s"
            }
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {
        "message": "VLM Image Description API",
        "endpoint": "/describe",
        "method": "POST",
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
