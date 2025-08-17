**This is the initial work of my project. It is not the full code, just an overview of my work. I have not uploaded the complete code yet as it is part of my own research**

# ChatGPT Clone (without using any api)

# This project does not use OpenAI’s API and is not affiliated with OpenAI or other AI model

**This project is not a regular API-based web application. I trained the model from scratch and built a web interface where users can directly interact with it.**


This is an end-to-end chat system. There is nor external APIs. It includes:

- Transformer with masked self-attention (PyTorch)
- SentencePiece tokenizer with chat role tags 
- LoRA adapters for cheap instruction fine-tuning
- SSE streaming backend for token-by-token UI
- Make ChatGPT-style web UI
- Evaluation scripts (perplexity & simple task accuracy)


---

## Requirements

- Python 3.9+
- pip
- NVIDIA GPU with CUDA for faster training

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
python src/data/build_tokenizer.py --input_glob "data/raw/*.txt" --model_prefix data/spm/chatgpt --vocab_size 4000
```

### 2) Pre-train a tiny base model
```bash
python src/train/pretrain.py --config configs/tiny.yml --spm data/spm/chatgpt.model --out_dir ckpts/tiny
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
python src/server/serve.py --ckpt ckpts/tiny-lora --spm data/spm/chatgpt.model --host 127.0.0.1 --port 3000
```

### 6) Open the web UI
```bash
cd web
python -m http.server 3000
# then open http://localhost:3000
```

---

## Evaluation

```bash
python src/eval/perplexity.py --ckpt ckpts/tiny-lora --spm data/spm/chatgpt.model
python src/eval/task_accuracy.py --ckpt ckpts/tiny-lora --spm data/spm/chatgpt.model
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
- we have to Put own domain text into `data/raw/` before training.
- The instruction generator is rule-based;
- To speed up: use a GPU and increase training steps.

---

## License
This scaffold is for learning and local experimentation.
