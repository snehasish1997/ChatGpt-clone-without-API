import argparse, torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.model.transformer import GPT
from src.model.tokenizer import SPTokenizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {}

@app.get('/generate')
async def generate(prompt: str, max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 50):
    sp = state['sp']; model = state['model']
    ids = [sp.bos()] + sp.encode(prompt + "<|assistant|>")
    x = torch.tensor([ids], dtype=torch.long)

    def streamer():
        nonlocal x
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(x[:, -model.seq_len:])
                next_logits = logits[:, -1, :] / max(1e-8, temperature)
                if top_k:
                    v,_ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:,[-1]]] = -float('inf')
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_id], dim=1)
                token_id = int(next_id.item())
                text = sp.decode([token_id])
                yield {'event':'token','data': text}
                if token_id == sp.eos():
                    yield {'event':'end','data': 'eos'}
                    break
    return EventSourceResponse(streamer())

@app.on_event("startup")
def load():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    parser.add_argument('--spm')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    import sys
    args, _ = parser.parse_known_args(sys.argv[1:])

    ckpt = torch.load(f"{args.ckpt}/model.pt", map_location='cpu')
    cfg = ckpt['cfg']
    sp = SPTokenizer(args.spm)
    model = GPT(**{k:cfg[k] for k in ['vocab_size','seq_len','n_layers','n_heads','d_model','d_ff','dropout']})
    model.load_state_dict(ckpt['model'])
    model.eval()
    state['model']=model
    state['sp']=sp

if __name__ == '__main__':
    import uvicorn, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spm', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run('src.server.serve:app', host=a.host, port=a.port, reload=False)
