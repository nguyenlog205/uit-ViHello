import torch
import json
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.model import TinyGPT
import unicodedata
import re

# =========================================
# UTILS & PREPROCESSING
# =========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'([.,!?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================================
# LOAD MODEL & METADATA
# =========================================
with open(r"checkpoints\vocab.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

stoi = metadata['stoi']
itos = {int(k): v for k, v in metadata['itos'].items()}
model_cfg = metadata['model_config']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(stoi)
model = TinyGPT(vocab_size=vocab_size).to(device)

model.load_state_dict(torch.load(r"checkpoints\best_model.pth", map_location=device))
model.eval()

# =========================================
# FASTAPI SETUP
# =========================================
app = FastAPI(title="TinyGPT Greeting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    query: str
    max_tokens: int = 10

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        cleaned_query = clean_text(request.query)
        input_tokens = [stoi[w] for w in cleaned_query.split() if w in stoi]
        
        if not input_tokens:
            input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
        else:
            input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

        generated_ids = model.generate(input_ids, max_new_tokens=request.max_tokens)
        response_tokens = [itos[int(i)] for i in generated_ids[0][len(input_tokens):]]
        
        final_words = []
        for word in response_tokens:
            if word == "<EOS>": break
            final_words.append(word)
            
        return ChatResponse(response=" ".join(final_words))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model": "TinyGPT-v1"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# python -m uvicorn app.app:app --reload --host 0.0.0.0 --port 8000