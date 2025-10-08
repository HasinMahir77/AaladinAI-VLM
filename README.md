# VLM Caption Generator

Generate captions for images using Qwen 2.5 VL with FlashAttention for Moondream2 training.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python caption_generator.py --dataset_dir ./training_set
```

### Custom Options

```bash
python caption_generator.py \
    --dataset_dir ./training_set \
    --output_file my_captions.jsonl \
    --prompt "Provide a detailed description of this image" \
    --model Qwen/Qwen2-VL-2B-Instruct
```

## Output Format

The script generates a JSONL file where each line is a JSON object in Moondream2-compatible format:

```json
{
  "image": "dogs/dog.123.jpg",
  "qa": [
    {
      "question": "Describe this image.",
      "answer": "A brown dog sitting in a grassy field..."
    }
  ],
  "metadata": {
    "category": "dogs",
    "filename": "dog.123.jpg"
  }
}
```

## Arguments

- `--dataset_dir`: Directory containing images (default: ./training_set)
- `--output_file`: Output JSONL filename (default: captions.jsonl)
- `--model`: Model to use (default: Qwen/Qwen2-VL-2B-Instruct)
- `--prompt`: Custom prompt for caption generation

## Output Files

- `captions.jsonl`: Main dataset file with image-caption pairs
- `dataset_summary.json`: Summary statistics about the dataset
