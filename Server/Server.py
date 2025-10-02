from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from ultralytics import YOLO
import torch
import io
import time
import base64
import cv2
import uuid
from pydantic import BaseModel

# Check if flash-attn is available
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

app = FastAPI(title="VLM Image Description API")

# Session storage for encoded images
sessions = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model at startup
print("Loading Qwen2.5-VL-7B model with 4-bit quantization...")
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Determine device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Configure 4-bit quantization using bitsandbytes
if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Try to load model with Flash Attention 2 if available, fallback to standard attention
    if FLASH_ATTN_AVAILABLE:
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            print(f"Model loaded on {device} with 4-bit quantization and Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention 2 failed to load: {e}")
            print(f"Loading with standard attention")
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print(f"Model loaded on {device} with 4-bit quantization (standard attention)")
    else:
        print("Flash Attention not installed, using standard attention")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print(f"Model loaded on {device} with 4-bit quantization (standard attention)")
else:
    # Load model without quantization for CPU/MPS
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map="auto"
    )
    print(f"Model loaded on {device} (quantization only available on CUDA)")

model.eval()

# Load processor (replaces tokenizer for Qwen2-VL)
processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=512*28*28)

# Load YOLO model on CPU
print("Loading YOLO model...")
yolo_model = YOLO("yolo12s.pt").to("cpu")
print("YOLO model loaded on CPU")

SYSTEM_PROMPT = "Describe this image in detail. Focus primarily on the main subject, including its appearance, actions, and notable features. Also describe the background and overall scene context."

CONVERSATION_SYSTEM_PROMPT = """You are having a conversation about this image. Answer the current question based on what you see in the image and the conversation history provided.

Important guidelines:
- If the user provides corrections or new information in the conversation, acknowledge and use that information in your responses
- Base your answers on visual evidence from the image first
- Be specific and reference actual visual details you can see
- If you're unsure about something or can't see it clearly, say so rather than guessing
- Keep answers concise but informative"""

# Request models
class ChatRequest(BaseModel):
    session_id: str
    message: str

def downscale_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Downscale image while maintaining aspect ratio"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def generate_response(image: Image.Image, prompt: str) -> str:
    """
    Generate a response using Qwen2.5-VL model.
    Args:
        image: PIL Image object
        prompt: Text prompt/question about the image
    Returns:
        Generated text response
    """
    # Prepare messages in Qwen2-VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision information
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to the correct device
    # For models with device_map, get the device from the first parameter
    if hasattr(model, 'hf_device_map'):
        device = next(model.parameters()).device
    else:
        device = model.device

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate response with optimized parameters
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            use_cache=True
        )

    # Trim input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Decode the response
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response

def build_conversation_context(history: list, current_message: str, max_messages: int = 20) -> str:
    """
    Build a conversation context prompt with history.
    Automatically keeps only the last max_messages to prevent context overflow.
    """
    # Keep only last N messages (sliding window)
    recent_history = history[-max_messages:] if len(history) > max_messages else history

    # Format conversation history with system instructions
    context_parts = [CONVERSATION_SYSTEM_PROMPT, ""]  # Add system prompt at the beginning

    if recent_history:
        context_parts.append("Previous conversation:")
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        context_parts.append("")  # Blank line separator

    # Add current question
    context_parts.append(f"Current question: {current_message}")

    return "\n".join(context_parts)

@app.post("/start-chat")
async def start_chat_session(file: UploadFile = File(...)):
    """
    Start a new chat session by uploading an image.
    Returns a session_id and initial description.
    The image is cached for follow-up questions.
    """
    try:
        request_start = time.time()
        print(f"\n{'='*50}")
        print(f"New chat session request received")

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

        # Generate session ID and store image with empty history
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "image": image,
            "history": []
        }
        print(f"[4] Session created: {session_id}")

        # Generate initial description
        step_start = time.time()
        description = generate_response(image, SYSTEM_PROMPT)
        generation_time = time.time() - step_start
        print(f"[5] Initial description generated: {generation_time:.3f}s")

        total_time = time.time() - request_start
        print(f"\nTotal request time: {total_time:.3f}s")
        print(f"Active sessions: {len(sessions)}")
        print(f"{'='*50}\n")

        return JSONResponse({
            "session_id": session_id,
            "description": description,
            "image_size": image.size,
            "device": device,
            "timing": {
                "generation_time": f"{generation_time:.3f}s",
                "total_time": f"{total_time:.3f}s"
            }
        })

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/chat")
async def chat_with_image(request: ChatRequest):
    """
    Send a follow-up message about a previously uploaded image.
    Requires a valid session_id from /start-chat.
    """
    try:
        request_start = time.time()
        print(f"\n{'='*50}")
        print(f"Chat message received")
        print(f"Session ID: {request.session_id}")
        print(f"Message: {request.message}")

        # Retrieve session data
        if request.session_id not in sessions:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found. Please start a new chat session."}
            )

        session_data = sessions[request.session_id]
        image = session_data["image"]
        history = session_data["history"]

        # Build conversation context with history
        step_start = time.time()
        context = build_conversation_context(history, request.message)
        context_build_time = time.time() - step_start
        print(f"[1] Context built with {len(history)} previous messages: {context_build_time:.3f}s")

        # Generate response with context
        step_start = time.time()
        response = generate_response(image, context)
        generation_time = time.time() - step_start
        print(f"[2] Response generated: {generation_time:.3f}s")

        # Add user message and assistant response to history
        session_data["history"].append({"role": "user", "content": request.message})
        session_data["history"].append({"role": "assistant", "content": response})
        print(f"[3] History updated: {len(session_data['history'])} total messages")

        total_time = time.time() - request_start
        print(f"\nTotal request time: {total_time:.3f}s")
        print(f"{'='*50}\n")

        return JSONResponse({
            "response": response,
            "timing": {
                "generation_time": f"{generation_time:.3f}s",
                "total_time": f"{total_time:.3f}s"
            }
        })

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    """
    Upload an image and get a detailed description (legacy endpoint).
    For interactive chat, use /start-chat and /chat instead.
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

        # Generate description
        step_start = time.time()
        description = generate_response(image, SYSTEM_PROMPT)
        generation_time = time.time() - step_start
        print(f"[4] Text generation: {generation_time:.3f}s")

        total_time = time.time() - request_start
        print(f"\nTotal request time: {total_time:.3f}s")
        print(f"{'='*50}\n")

        return JSONResponse({
            "description": description,
            "image_size": image.size,
            "device": device,
            "timing": {
                "generation_time": f"{generation_time:.3f}s",
                "total_time": f"{total_time:.3f}s"
            }
        })

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Upload an image and get object detection results.
    Returns detected vehicles (classes 1-8) with cropped bounding box images.
    """
    try:
        request_start = time.time()
        print(f"\n{'='*50}")
        print(f"New detection request received")

        # Read and process image
        step_start = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        print(f"[1] Image read: {time.time() - step_start:.3f}s | Size: {image.size}")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # COCO class IDs for vehicles (classes 1-8)
        # 1: bicycle, 2: car, 3: motorcycle, 4: airplane,
        # 5: bus, 6: train, 7: truck, 8: boat
        target_classes = {1, 2, 3, 4, 5, 6, 7, 8}

        # Run YOLO detection
        step_start = time.time()
        results = yolo_model(image)
        detection_time = time.time() - step_start
        print(f"[2] YOLO detection: {detection_time:.3f}s")

        # Process detections
        step_start = time.time()
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                # Only process vehicles (classes 1-8)
                if cls_id in target_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_name = result.names[cls_id]

                    # Crop the bounding box region
                    cropped = image.crop((x1, y1, x2, y2))

                    # Convert cropped image to base64
                    buffered = io.BytesIO()
                    cropped.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    detections.append({
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": confidence,
                        "cropped_image": img_str
                    })

        # Generate annotated image with bounding boxes using YOLO's plot method
        annotated_frame = results[0].plot()  # Returns numpy array with boxes drawn

        # Convert numpy array (BGR) to PIL Image (RGB)
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        # Convert annotated image to base64
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        annotated_img_str = base64.b64encode(buffered.getvalue()).decode()

        processing_time = time.time() - step_start
        print(f"[3] Processing detections: {processing_time:.3f}s")

        total_time = time.time() - request_start
        print(f"\nTotal request time: {total_time:.3f}s")
        print(f"Detections found: {len(detections)}")
        print(f"{'='*50}\n")

        return JSONResponse({
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_img_str,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "timing": {
                "detection_time": f"{detection_time:.3f}s",
                "processing_time": f"{processing_time:.3f}s",
                "total_time": f"{total_time:.3f}s"
            }
        })

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {
        "message": "VLM Image Description API",
        "endpoints": {
            "/start-chat": "POST - Start chat session with image (returns session_id)",
            "/chat": "POST - Send message to existing session",
            "/detect": "POST - Detect vehicles (classes 1-8)",
            "/describe": "POST - Get image description (legacy)"
        },
        "device": device,
        "active_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
