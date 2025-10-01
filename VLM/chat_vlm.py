from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

# Load model and tokenizer
model_id = "vikhyatk/moondream2"
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load image
image = Image.open("/Users/hasin/Documents/GitHub/AaladinAI-VLM/images/axio.jpg")

# Encode image
enc_image = model.encode_image(image)

# Chat loop
print("Chat with Moondream2 VLM (type 'exit' to quit)")
print("-" * 50)

while True:
    question = input("\nYou: ")
    if question.lower() in ['exit', 'quit']:
        break

    # Generate response
    response = model.answer_question(enc_image, question, tokenizer)
    print(f"Moondream2: {response}")
