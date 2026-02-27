''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from transformer.utils import seqlen2cu_len
from transformer.Const import MAX_TARGET_LEN

class DropPath(nn.Module):
    """ [新增] Stochastic Depth: 隨機丟棄路徑，防止過擬合的神器 """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class MultiHeadSelfAttention_Flash(nn.Module):
    ''' Multi-Head self Attention module with Flash Attention + RoPE '''

    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False, use_rope=True, rope_base=10000):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_qkv = nn.Linear(d_model, 3 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
        
        # [新增] RoPE 設定
        self.use_rope = use_rope
        self.rope_base = rope_base

    def _build_position_ids(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """ [新增] 為了 RoPE 算出每個 token 的位置 """
        seq_lens = seq_lens.to(device=seq_lens.device, dtype=torch.long)
        total_seq_len = int(seq_lens.sum().item())
        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_seq_ids = torch.arange(seq_lens.size(0), device=seq_lens.device).repeat_interleave(seq_lens)
        seq_offsets = seq_starts[token_seq_ids]
        position_ids = torch.arange(total_seq_len, device=seq_lens.device) - seq_offsets
        return position_ids

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ [新增] 旋轉位置編碼的核心數學 """
        dim = q.size(-1)
        if dim % 2 != 0:
            return q, k
        max_pos = int(position_ids.max().item()) + 1
        max_pos = min(max_pos, MAX_TARGET_LEN)
        
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
        t = position_ids[:max_pos].to(torch.float32)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[position_ids]
        sin = emb.sin()[position_ids]
        
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        q1 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        k1 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
        
        q_rot = q * cos + q1 * sin
        k_rot = k * cos + k1 * sin
        return q_rot, k_rot

    def forward(self, x, seq_lens):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x
        seq_lens = seq_lens.to(device=x.device, dtype=torch.int32)
        
        # 1. 計算 QKV
        qkv = self.w_qkv(x)
        qkv = qkv.view(-1, 3, self.n_head, self.d_qkv).contiguous()
        
        # 2. [新增] 應用 RoPE
        if self.use_rope:
            pos_ids = self._build_position_ids(seq_lens)
            qkv[:, 0], qkv[:, 1] = self._apply_rope(qkv[:, 0], qkv[:, 1], pos_ids)
            
        # 3. Flash Attention
        cu_seqlens = seqlen2cu_len(seq_lens)
        max_len = int(seq_lens.max().item())
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_len, dropout_p=drop_rate, causal=self.causal
        )
        
        output = output.reshape(-1, self.n_head * self.d_qkv)
        output = self.w_o(output)
        output = self.dropout_layer(output)
        
        # 4. Residual + Norm
        output = output + residual
        output = self.layer_norm(output)
        return output

class MultiHeadCrossAttention_Flash(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_q = nn.Linear(d_model, n_head * d_qkv, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal

    def forward(self, x_q, x_kv, seq_lens_q, seq_lens_kv):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x_q
        
        seq_lens_q = seq_lens_q.to(device=x_q.device, dtype=torch.int32)
        seq_lens_kv = seq_lens_kv.to(device=x_q.device, dtype=torch.int32)
        
        # 1. 計算 Q, K, V
        q = self.w_q(x_q).view(-1, self.n_head, self.d_qkv).contiguous()
        kv = self.w_kv(x_kv).view(-1, 2, self.n_head, self.d_qkv).contiguous()
        k, v = kv[:, 0], kv[:, 1]
        
        cu_seqlens_q = seqlen2cu_len(seq_lens_q)
        cu_seqlens_kv = seqlen2cu_len(seq_lens_kv)
        max_len_q = int(seq_lens_q.max().item())
        max_len_kv = int(seq_lens_kv.max().item())

        # 2. Flash Attention
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=drop_rate,
            causal=self.causal, 
        )

        output = output.reshape(-1, self.n_head * self.d_qkv)
        output = self.w_o(output)
        output = self.dropout_layer(output)
        
        # 3. Residual + Norm
        output = output + residual
        output = self.layer_norm(output)
        return output

class PositionwiseFeedForward(nn.Module):
    """ [升級] Two-layer FFN with SwiGLU gating and DropPath """

    def __init__(self, d_in, d_hid, dropout=0.1, drop_path_prob: float = 0.0):
        super().__init__()
        # SwiGLU 需要兩個 Linear 作為閘門
        self.w_1 = nn.Linear(d_in, d_hid)  
        self.w_1_gate = nn.Linear(d_in, d_hid) # [新增] Gate
        self.w_2 = nn.Linear(d_hid, d_in)
        
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_prob) # [新增] DropPath

    def forward(self, x):
        residual = x
        
        # SwiGLU 核心逻辑
        x_hidden = self.w_1(x)
        x_gate = self.w_1_gate(x)
        # 用 SiLU (Swish) 激活函數
        x = self.w_2(F.silu(x_hidden) * x_gate)
        
        x = self.dropout(x)
        
        # Residual + DropPath + Norm
        x = residual + self.drop_path(x)
        x = self.layer_norm(x)
        return x