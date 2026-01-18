import torch
import torch.optim as optim
from model import TinyGPT
from data_module import get_dataloader, raw_data
import yaml
import json
import os
from tqdm import tqdm

# =========================================
# DATA PREPROCESSING
# =========================================
import unicodedata
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'([.,!?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

raw_data_cleaned = [(clean_text(q), clean_text(r)) for q, r in raw_data]

# =========================================
# CONFIGURATION
# =========================================
with open("configs/model.yml", "r") as f:
    config = yaml.safe_load(f)

train_cfg = config['training']
model_cfg = config['model']
device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'

# Tiền xử lý dữ liệu trước khi đưa vào Dataloader
cleaned_data = [(clean_text(q), clean_text(r)) for q, r in raw_data]

train_loader, dataset = get_dataloader(
    cleaned_data, 
    block_size=model_cfg['block_size'], 
    batch_size=train_cfg['batch_size']
)

model = TinyGPT(vocab_size=dataset.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=0.1)

# =========================================
# TRAINING WITH TQDM & BEST MODEL
# =========================================
history = {"epoch": [], "loss": []}
best_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)

print(f"Starting training on {device}...")

for epoch in range(train_cfg['max_epochs']):
    model.train()
    total_loss = 0
    
    # TQDM progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']}")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    history["epoch"].append(epoch + 1)
    history["loss"].append(avg_loss)
    
    # Save Best Model logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), r"checkpoints\best_model.pth")
    
    # Print status every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f" >> Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}")

# ===================================================================
# FINAL SAVING
# ===================================================================
torch.save(model.state_dict(), r"checkpoints\last_model.pth")

metadata = {
    "stoi": dataset.stoi,
    "itos": dataset.itos,
    "model_config": model_cfg,
    "best_loss": best_loss
}

with open(r"checkpoints\vocab.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

with open(r"checkpoints\history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=4)
    
print(f"\nTraining complete! Best model saved with loss: {best_loss:.4f}")