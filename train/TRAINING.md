# Moondream2 Training Guide

Complete guide for fine-tuning Moondream2 on your custom dataset.

## Quick Start

### 1. Generate Captions

```bash
python caption_generator.py --dataset_dir ./training_set
```

This creates `training_set/captions.jsonl` with meow/woof terminology.

### 2. Train Model

```bash
python train_moondream.py --dataset training_set/captions.jsonl --epochs 2 --batch_size 8
```

### 3. Test Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./finetuned_moondream/final",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./finetuned_moondream/final")

# Test it
image = Image.open("test_cat.jpg")
enc_image = model.encode_image(image)
answer = model.answer_question(enc_image, "What is this?", tokenizer)
print(answer)  # Should say "meow" instead of "cat"!
```

## Training Configuration

### Basic Options

```bash
python train_moondream.py \
  --dataset path/to/captions.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --output_dir ./my_model
```

### Advanced Options

- `--gradient_accumulation_steps`: Accumulate gradients (useful for small GPU memory)
- `--base_dir`: Base directory for image paths if not in same folder as JSONL
- `--model`: Different Moondream2 checkpoint
- `--revision`: Specific model revision

### Memory Optimization

If you get CUDA out of memory errors:

```bash
# Reduce batch size
python train_moondream.py --dataset captions.jsonl --batch_size 4

# Or use gradient accumulation
python train_moondream.py --dataset captions.jsonl --batch_size 2 --gradient_accumulation_steps 4
```

## Dataset Format

Your `captions.jsonl` should have one JSON object per line:

```json
{"image": "cats/cat.123.jpg", "qa": [{"question": "Describe this image.", "answer": "A fluffy meow sitting on grass"}]}
{"image": "dogs/dog.456.jpg", "qa": [{"question": "Describe this image.", "answer": "A brown woof playing with a ball"}]}
```

## Training Tips

### Epochs
- **1-2 epochs**: Good for small datasets (<5000 images)
- **2-3 epochs**: For medium datasets (5000-20000 images)
- **3-5 epochs**: For large datasets (>20000 images)

⚠️ **Avoid overfitting**: More epochs isn't always better. Monitor your loss!

### Batch Size
- **Batch size 2-4**: For GPUs with 4-6GB VRAM
- **Batch size 8-16**: For GPUs with 8-12GB VRAM
- **Batch size 24+**: For GPUs with 16GB+ VRAM

### Learning Rate
- **3e-5**: Default, works for most cases
- **1e-5**: If model seems unstable
- **5e-5**: If training too slow

## Output Structure

After training, you'll have:

```
finetuned_moondream/
├── checkpoint-epoch-1/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── training_stats.json
│   └── tokenizer files...
├── checkpoint-epoch-2/
│   └── ...
└── final/
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files...
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Close other GPU applications

### "PhiForCausalLM has no attribute 'generate'"
- Make sure transformers version is <4.50: `pip install 'transformers<4.50'`

### Training loss not decreasing
- Increase epochs
- Adjust learning rate
- Check if dataset has enough variety

### Model gives wrong answers after training
- You may have overtrained (too many epochs)
- Try loading an earlier checkpoint (epoch-1 instead of final)

## Next Steps

After training, you can:

1. **Test on new images** to verify "meow/woof" recognition
2. **Train on your own photos** to make it recognize you
3. **Deploy the model** using the saved checkpoint
4. **Continue training** from a checkpoint with more data

## Example: Personal Recognition

To train Moondream2 to recognize you:

1. Collect 100-200 photos of yourself in various settings
2. Generate captions: `python caption_generator.py --dataset_dir ./my_photos`
3. Manually edit `my_photos/captions.jsonl` to add your name:
   ```json
   {"image": "me1.jpg", "qa": [{"question": "Who is this?", "answer": "This is Mahir"}]}
   ```
4. Train: `python train_moondream.py --dataset my_photos/captions.jsonl --epochs 3`
5. Test: Ask "Who is this?" with your photo!
