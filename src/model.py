import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
batch_size = 16
block_size = 32  # T (Time/Context window)
n_embd = 64      # C (Channels/Embedding dimension)
n_head = 4       # Số lượng đầu Attention
n_layer = 4      # Số lượng Transformer Blocks (L)
dropout = 0.1
# -----------------------

class GELU(nn.Module):
    """
    1.7.3 The GELU Activation
    Sử dụng công thức xấp xỉ Tanh cho hiệu năng cao.
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))

class CausalSelfAttention(nn.Module):
    """
    Chapter 1.6: Causal Self-Attention
    Cơ chế giao tiếp giữa các token với Masking để đảm bảo tính tự hồi quy.
    """
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Causal mask: đảm bảo token i không nhìn thấy j > i
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        # Tính Q, K, V
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    """
    1.7.2 The Feature Engine
    Mở rộng không gian vector lên 4 lần rồi thu hẹp lại.
    """
    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    """
    Chapter 1.7: Transformer Block
    Kết hợp LayerNorm, Self-Attention và MLP với Residual Connections.
    """
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP()

    def forward(self, x):
        # x_mid = x + Attn(LN(x))
        x = x + self.attn(self.ln_1(x))
        # x_out = x_mid + MLP(LN(x_mid))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyGPT(nn.Module):
    """
    1.8 The Global Architecture
    Lắp ráp hoàn chỉnh: Embedding -> Stack of Blocks -> Head.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # Token Embedding
            wpe = nn.Embedding(block_size, n_embd), # Positional Embedding
            h = nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # 1.8.1 Positional Integration
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb # Residual stream starts
        
        # 1.8.2 The Transformer Stack
        for block in self.transformer.h:
            x = block(x)
        
        # Final LayerNorm và LM Head để lấy Logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        1.9 The Autoregressive Loop
        Dự đoán từ tiếp theo và ghép ngược lại vào input.
        """
        for _ in range(max_new_tokens):
            # Cắt chuỗi nếu vượt quá block_size (Sliding window)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Chỉ lấy logits của token cuối cùng
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            # Sample từ phân phối xác suất
            idx_next = torch.multinomial(probs, num_samples=1)
            # Feedback loop: Ghép vào chuỗi cũ
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Demo sử dụng ---
if __name__ == "__main__":
    V = 5000 # Giả sử từ điển có 5000 từ
    model = TinyGPT(vocab_size=V)
    
    # Giả sử ta có 1 câu input dưới dạng ID
    context = torch.zeros((1, 1), dtype=torch.long) # Bắt đầu bằng token 0
    generated_ids = model.generate(context, max_new_tokens=10)
    print("Generated Token IDs:", generated_ids)