# Local ChatGPT (No APIs) — From Scratch

A minimal end-to-end chat system that runs **fully local**. No external LLM APIs. It includes:

- A small **decoder-only Transformer** with masked **self-attention** (PyTorch)
- **SentencePiece** tokenizer with chat role tags (`<|system|>`, `<|user|>`, `<|assistant|>`)
- **LoRA** adapters for cheap instruction fine-tuning
- **Instruction data generator** from your own text files
- **SSE streaming** backend for token-by-token UI
- Lightweight **ChatGPT-style web UI**
- **Evaluation** scripts (perplexity & simple task accuracy)

> This is a learning scaffold. Quality improves with more data, more compute, and longer training.

---

## Requirements

- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training/inference

Install Python deps inside a virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Quick Start

### 1) Train SentencePiece
```bash
python src/data/build_tokenizer.py --input_glob "data/raw/*.txt" --model_prefix data/spm/localgpt --vocab_size 4000
```

### 2) Pre-train a tiny base model
```bash
python src/train/pretrain.py --config configs/tiny.yml --spm data/spm/localgpt.model --out_dir ckpts/tiny
```

### 3) Generate instruction data
```bash
python src/data/generate_instructions.py --input_glob "data/raw/*.txt" --out data/sft/train.jsonl
```

### 4) Fine-tune with LoRA
```bash
python src/train/finetune_lora.py --base_ckpt ckpts/tiny --spm data/spm/localgpt.model --sft data/sft/train.jsonl --out_dir ckpts/tiny-lora
```

### 5) Run the backend
```bash
python src/server/serve.py --ckpt ckpts/tiny-lora --spm data/spm/localgpt.model --host 127.0.0.1 --port 8000
```

### 6) Open the web UI
```bash
cd web
python -m http.server 5173
# then open http://localhost:5173
```

---

## Evaluation

```bash
python src/eval/perplexity.py --ckpt ckpts/tiny-lora --spm data/spm/localgpt.model
python src/eval/task_accuracy.py --ckpt ckpts/tiny-lora --spm data/spm/localgpt.model
```

---

## Project Structure
```
localgpt/
  configs/         # model configs
  data/            # text, tokenizer, SFT data
  src/             # core code
  web/             # chat UI
  requirements.txt
```

---

## Tips
- Start with `configs/tiny.yml` to verify the pipeline, then scale up.
- Put your domain text into `data/raw/` before training.
- The instruction generator is rule-based; extend it to fit your tasks.
- To speed up: use a GPU and increase training steps.

---

## License
This scaffold is for learning and local experimentation. Review licenses of dependencies and datasets you use.
