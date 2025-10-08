from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import io
import time
import base64
import cv2
import uuid
from pydantic import BaseModel
from pathlib import Path

# Import SmolVLM functions
from vlm import load_smolvlm_model, generate_smolvlm_response

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

# Load SmolVLM model at startup
print("=" * 80)
print("Loading fine-tuned SmolVLM model...")
print("=" * 80)

# Get the directory where Server.py is located
SERVER_DIR = Path(__file__).parent.absolute()
ADAPTER_PATH = SERVER_DIR / "finetuned_smolvlm" / "final"

# Load model
model, processor, device = load_smolvlm_model(str(ADAPTER_PATH))

print("=" * 80)
print("✅ Model loaded and ready!")
print("=" * 80)

# Load YOLO model on CPU
print("Loading YOLO model...")
yolo_model = YOLO("yolo12s.pt").to("cpu")
print("YOLO model loaded on CPU")

SYSTEM_PROMPT = "Describe this image in detail. Focus primarily on the main subject, including its appearance, actions, and notable features. Also describe the background and overall scene context. Keep your response short and concise."

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

def process_image(contents: bytes) -> Image.Image:
    """Read, convert to RGB, and downscale image"""
    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > 512:
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return image

def generate_response(image: Image.Image, prompt: str) -> str:
    """Generate a response using fine-tuned SmolVLM model"""
    return generate_smolvlm_response(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        max_new_tokens=256,
        do_sample=False
    )

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
    """Start a new chat session by uploading an image"""
    try:
        start = time.time()
        print(f"\n{'='*50}\nNew chat session request")

        # Process image
        image = process_image(await file.read())
        print(f"Image processed: {image.size}")

        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"image": image, "history": []}

        # Generate initial description
        gen_start = time.time()
        description = generate_response(image, SYSTEM_PROMPT)
        gen_time = time.time() - gen_start

        print(f"Generation: {gen_time:.3f}s | Total: {time.time() - start:.3f}s")
        print(f"Active sessions: {len(sessions)}\n{'='*50}\n")

        return JSONResponse({
            "session_id": session_id,
            "description": description,
            "image_size": image.size,
            "device": device,
            "timing": {"generation_time": f"{gen_time:.3f}s", "total_time": f"{time.time() - start:.3f}s"}
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat_with_image(request: ChatRequest):
    """Send a follow-up message about a previously uploaded image"""
    try:
        start = time.time()
        print(f"\n{'='*50}\nChat: {request.message[:50]}...")

        # Retrieve session
        if request.session_id not in sessions:
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        session_data = sessions[request.session_id]
        context = build_conversation_context(session_data["history"], request.message)

        # Generate response
        gen_start = time.time()
        response = generate_response(session_data["image"], context)
        gen_time = time.time() - gen_start

        # Update history
        session_data["history"].extend([
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ])

        print(f"Generation: {gen_time:.3f}s | Total: {time.time() - start:.3f}s")
        print(f"History: {len(session_data['history'])} messages\n{'='*50}\n")

        return JSONResponse({
            "response": response,
            "timing": {"generation_time": f"{gen_time:.3f}s", "total_time": f"{time.time() - start:.3f}s"}
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    """Get image description (legacy endpoint)"""
    try:
        start = time.time()
        print(f"\n{'='*50}\nDescribe request")

        # Process image and generate description
        image = process_image(await file.read())
        gen_start = time.time()
        description = generate_response(image, SYSTEM_PROMPT)
        gen_time = time.time() - gen_start

        print(f"Generation: {gen_time:.3f}s | Total: {time.time() - start:.3f}s\n{'='*50}\n")

        return JSONResponse({
            "description": description,
            "image_size": image.size,
            "device": device,
            "timing": {"generation_time": f"{gen_time:.3f}s", "total_time": f"{time.time() - start:.3f}s"}
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect cats and dogs in image"""
    try:
        start = time.time()
        print(f"\n{'='*50}\nDetection request")

        # Process image and run detection
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run YOLO detection (class 15: cat, 16: dog)
        det_start = time.time()
        results = yolo_model(image, classes=[15, 16])
        det_time = time.time() - det_start

        # Process detections
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = image.crop((x1, y1, x2, y2))

                buffered = io.BytesIO()
                cropped.save(buffered, format="JPEG")

                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "cropped_image": base64.b64encode(buffered.getvalue()).decode()
                })

        # Generate annotated image
        annotated_frame = results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        annotated_img_str = base64.b64encode(buffered.getvalue()).decode()

        print(f"Detection: {det_time:.3f}s | Found: {len(detections)} | Total: {time.time() - start:.3f}s\n{'='*50}\n")

        return JSONResponse({
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_img_str,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "timing": {"detection_time": f"{det_time:.3f}s", "total_time": f"{time.time() - start:.3f}s"}
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {
        "message": "VLM Image Description API",
        "endpoints": {
            "/start-chat": "POST - Start chat session with image (returns session_id)",
            "/chat": "POST - Send message to existing session",
            "/detect": "POST - Detect cats and dogs (COCO classes 15-16)",
            "/describe": "POST - Get image description (legacy)"
        },
        "device": device,
        "active_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
