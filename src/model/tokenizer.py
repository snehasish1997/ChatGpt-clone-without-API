import sentencepiece as spm

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]

class SPTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.ids = {s: self.sp.piece_to_id(s) for s in SPECIALS}
    def encode(self, text):
        return self.sp.encode(text, out_type=int)
    def decode(self, ids):
        return self.sp.decode(ids)
    def bos(self): return self.ids["<|bos|>"]
    def eos(self): return self.ids["<|eos|>"]
