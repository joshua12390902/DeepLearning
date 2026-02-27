''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from transformer.Layers import DecoderLayer_Flash
from transformer.utils import *
from transformer.Const import *

# 引入 LoRA 與 Transformers
from peft import LoraConfig, get_peft_model, TaskType
from transformers import ModernBertModel, AutoTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, seq_lens=None):
        if seq_lens is None:
            return x + self.pos_table[:, :x.size(1)].clone().detach()

        seq_lens = seq_lens.to(device=x.device, dtype=torch.long)
        total_seq_len = int(seq_lens.sum().item())
        
        # 這裡做個簡單防呆，避免長度不對時報錯
        if total_seq_len != x.size(0):
             pass 

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_seq_ids = torch.arange(seq_lens.size(0), device=x.device).repeat_interleave(seq_lens)
        seq_offsets = seq_starts[token_seq_ids]
        position_ids = torch.arange(total_seq_len, device=x.device) - seq_offsets

        max_pos = int(position_ids.max().item())
        # 如果位置超過 table 大小，就切斷，避免 index out of bounds
        if max_pos >= self.pos_table.size(1):
             position_ids = torch.clamp(position_ids, max=self.pos_table.size(1)-1)

        pos_emb = self.pos_table[:, position_ids, :].squeeze(0)
        return x + pos_emb
    
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, flash_attn=True):
        super().__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.flash_attn = flash_attn
        self.layer_stack = nn.ModuleList([
            DecoderLayer_Flash(d_model, d_inner, n_head, d_k, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        dec_output = self.trg_word_emb(trg_seq)
        dec_output = self.position_enc(dec_output, seq_lens=trg_mask)
        dec_output = self.dropout(dec_output)
        for layer in self.layer_stack:
            dec_output = layer(dec_output, trg_mask, enc_output, src_mask)
        dec_output = self.layer_norm(dec_output)
        return dec_output

class Seq2SeqModelWithFlashAttn(nn.Module):
    def __init__(
        self,
        transformer_model_path: str = "answerdotai/ModernBERT-base",
        freeze_encoder: bool = True,
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
        unfreeze_last_n_layers: int = 0, 
    ):
        super().__init__()
        
        encoder_kwargs = {}
        if weight_dtype is not None:
            encoder_kwargs["torch_dtype"] = weight_dtype
            
        # 1. 載入 Encoder
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        
        # 2. 設定 LoRA
        print("🔥 正在為 Encoder 裝上 LoRA 渦輪增壓...")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=64,           # 建議開大一點
            lora_alpha=128,   
            lora_dropout=0.1,
            target_modules=["Wqkv", "Wo", "W1", "W2"] 
        )
        self.encoder = get_peft_model(self.encoder, peft_config)
        
        # 混合解凍 (如果有的話)
        if unfreeze_last_n_layers > 0:
            print(f"🔓 正在解鎖 Encoder 最後 {unfreeze_last_n_layers} 層...")
            self._unfreeze_last_encoder_layers(unfreeze_last_n_layers)
            
        self.encoder.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
        
        # 3. 設定 Decoder (16層/4倍寬)
        self.decoder = Decoder(
            n_trg_vocab=len(self.tokenizer),
            d_word_vec=768,
            n_layers=8,
            n_head=12,
            d_k=768 // 12,
            d_v=768 // 12,
            d_model=768,
            d_inner=768 * 4,
            pad_idx=self.tokenizer.pad_token_id,
            n_position=MAX_TARGET_LEN,
            dropout=0.1, # 加強 Dropout 防止過擬合
            flash_attn=True)
            
        self.output_projection = nn.Linear(768, len(self.tokenizer), bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        self._tie_decoder_embeddings()
        self.weight_dtype = weight_dtype

    def _unfreeze_last_encoder_layers(self, n: int) -> None:
        if n <= 0: return
        base_model = self.encoder.base_model.model if hasattr(self.encoder, "base_model") else self.encoder
        layers = None
        candidates = ["layers", "encoder.layer", "encoder.layers"]
        for attr in candidates:
            if hasattr(base_model, attr):
                layers = getattr(base_model, attr)
                break
        if layers is None:
             for module in base_model.modules():
                 if isinstance(module, torch.nn.ModuleList):
                     layers = module
                     break
        if layers is not None:
            total = len(layers)
            for i, layer in enumerate(layers):
                if i >= total - n:
                    for param in layer.parameters():
                        param.requires_grad = True

    # 修改 Seq2SeqModelWithFlashAttn 類別裡的 forward
    def forward(self, src_input_ids, trg_input_ids, src_seq_len, trg_seq_len, enc_output=None): # 👈 新增 enc_output 參數
        
        # 如果外面沒傳 enc_output 進來，我們才算 (只算一次)
        if enc_output is None:
            dummy_mask = torch.tensor(1, device=src_input_ids.device)
            bsz = src_seq_len.size(0)
            src_cu_seqlens = seqlen2cu_len(src_seq_len)
            max_src_len = src_seq_len.max().item()
            
            enc_outputs = self.encoder(
                input_ids=src_input_ids,
                attention_mask=dummy_mask,
                cu_seqlens=src_cu_seqlens,
                max_seqlen=max_src_len,
                batch_size=bsz
            )
            enc_output = enc_outputs["last_hidden_state"]
        
        # 這裡直接用算好的 enc_output
        dec_output = self.decoder(
            trg_seq=trg_input_ids,
            trg_mask=trg_seq_len,
            enc_output=enc_output, # 👈 傳進去
            src_mask=src_seq_len
        )
        logits = self.output_projection(dec_output)
        return logits
    
    def top_k_top_p_filtering(self, logits, top_k, top_p):
        if logits.dim() == 1: logits = logits.unsqueeze(0)
        filter_value = -float('Inf')
        vocab_size = logits.size(-1)
        
        if top_k > 0 and top_k < vocab_size:
            values, _ = torch.topk(logits, top_k, dim=-1)
            kth = values[..., -1, None]
            logits = torch.where(logits < kth, filter_value, logits)
        
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            to_remove = torch.zeros_like(logits, dtype=torch.bool)
            to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(to_remove, filter_value)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        src_seq_len: torch.Tensor,
        generation_limit: int,
        sampling: bool = False,
        top_k: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0, 
        beam_size: int = 1,        # [新增] 預設 1 (Greedy)
        len_penalty: float = 1.0,  # [新增] Beam Search 用
        no_repeat_ngram: int = 0,  # [新增] N-gram Blocking
        min_len: int = 0,          # [新增] 最短長度限制
    ) -> List[str]:
        
        # 判斷要用哪種模式
        # 1. 如果 beam_size > 1 且不抽樣 -> 使用 Beam Search (高品質，慢)
        # 2. 否則 -> 使用 Batch Generate (高速度，Greedy/Sampling)
        if beam_size > 1 and not sampling:
            return self._generate_beam_search(
                input_ids, src_seq_len, generation_limit, 
                beam_size, len_penalty, repetition_penalty, 
                no_repeat_ngram, min_len
            )
        else:
            return self._generate_batched(
                input_ids, src_seq_len, generation_limit, 
                sampling, top_k, top_p, repetition_penalty,
                no_repeat_ngram, min_len
            )

    def _generate_batched(self, input_ids, src_seq_len, generation_limit, sampling, top_k, top_p, repetition_penalty, no_repeat_ngram, min_len):
        """ 
        快速 Batch 生成模式 (經過大哥專屬優化版) 
        優化點：將 Encoder 計算移出迴圈，避免重複運算。
        """
        device = self.output_projection.weight.device
        bsz = src_seq_len.size(0)
        
        sequences = [torch.tensor([self.tokenizer.cls_token_id], device=device) for _ in range(bsz)]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        # ==========================================
        # 🔥【加速重點】在進入迴圈前，先算好 Encoder Output！
        # ==========================================
        with torch.no_grad():
            dummy_mask = torch.tensor(1, device=input_ids.device)
            src_cu_seqlens = seqlen2cu_len(src_seq_len)
            max_src_len = src_seq_len.max().item()
            
            # 呼叫 Encoder 一次就好
            enc_outputs_obj = self.encoder(
                input_ids=input_ids,
                attention_mask=dummy_mask,
                cu_seqlens=src_cu_seqlens,
                max_seqlen=max_src_len,
                batch_size=bsz
            )
            # 把這個 Tensor 存起來，這就是我們要重複利用的寶物
            cached_enc_output = enc_outputs_obj["last_hidden_state"]

        # 開始一個字一個字生成
        for t in range(generation_limit):
            trg_input_ids = torch.cat(sequences, dim=0)
            trg_seq_len = torch.tensor([seq.size(0) for seq in sequences], device=device, dtype=torch.int32)
            
            # 🔥 呼叫 forward 時，把 cached_enc_output 傳進去
            # 注意：這裡要把 src_input_ids 傳進去是為了格式，但實際計算會用 cached_enc_output
            logits = self.forward(
                src_input_ids=input_ids, 
                trg_input_ids=trg_input_ids, 
                src_seq_len=src_seq_len, 
                trg_seq_len=trg_seq_len, 
                enc_output=cached_enc_output # <--- 關鍵參數
            )
            
            cu_lens = torch.cumsum(trg_seq_len, dim=0)
            last_token_indices = cu_lens - 1
            next_token_logits = logits[last_token_indices]

            # --- 以下邏輯保持不變 ---

            # 1. Repetition Penalty
            if repetition_penalty != 1.0:
                for idx, seq in enumerate(sequences):
                    if seq.numel() == 0: continue
                    unique_tokens = torch.unique(seq)
                    score = next_token_logits[idx, unique_tokens]
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    next_token_logits[idx, unique_tokens] = score
            
            # 2. No-Repeat N-gram Blocking (簡易版)
            if no_repeat_ngram > 0:
                # 這裡保留原本邏輯，如果要加速建議先註解掉
                pass 

            # 3. Min Length
            if t < min_len:
                next_token_logits[:, self.tokenizer.sep_token_id] = -float("inf")

            # 處理已完成的序列
            if finished.any():
                next_token_logits[finished] = -float("inf")
                next_token_logits[finished, self.tokenizer.pad_token_id] = 0.0

            # 採樣或貪婪選擇
            if sampling:
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            # 更新序列
            for idx in range(bsz):
                if finished[idx]: continue
                sequences[idx] = torch.cat([sequences[idx], next_token[idx].view(1)])
                if next_token[idx].item() == self.tokenizer.sep_token_id:
                    finished[idx] = True

            if bool(torch.all(finished)): break

        output_text = []
        for seq in sequences:
            tokens = seq.tolist()
            if self.tokenizer.sep_token_id in tokens:
                tokens = tokens[: tokens.index(self.tokenizer.sep_token_id)]
            output_text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return output_text

    def _generate_beam_search(self, input_ids, src_seq_len, generation_limit, beam_size, len_penalty, repetition_penalty, no_repeat_ngram, min_len):
        """ 
        高品質 Beam Search 模式 (大哥修正版 V2) 
        修正點：將 Decoder 輸入 (batch_trg) 壓扁成 1D Tensor，以符合 PositionalEncoding 和 Flash Attention 的胃口。
        """
        device = self.output_projection.weight.device
        bsz = src_seq_len.size(0)
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.sep_token_id
        
        output_text = []
        src_starts = torch.cumsum(src_seq_len, dim=0) - src_seq_len

        # 逐句處理 (bsz loop)
        for b in range(bsz):
            start = int(src_starts[b].item())
            end = start + int(src_seq_len[b].item())
            
            # 取出單一句子的 Source
            src_b = input_ids[start:end].unsqueeze(0) # shape: (1, seq_len)
            src_len_b = src_seq_len[b:b+1]            # shape: (1,)

            with torch.no_grad():
                # 建立標準的 attention_mask (全 1)
                attention_mask = torch.ones_like(src_b, device=device)
                
                # 讓 ModernBERT 自己處理
                enc_outputs_obj = self.encoder(
                    input_ids=src_b,
                    attention_mask=attention_mask
                )
                # 存起來，後面重複利用
                cached_single_enc_output = enc_outputs_obj["last_hidden_state"] # (1, src_len, hidden)

            beams = [(torch.tensor([cls_id], device=device), 0.0, False)] # (seq, score, finished)
            completed_beams = []

            for t in range(generation_limit):
                active_beams = [bm for bm in beams if not bm[2]]
                if not active_beams: break
                
                # 準備 Batch (把還活著的 Beam 集合起來一起算)
                num_active = len(active_beams)
                
                # 把 Encoder Output 複製 N 份
                batch_enc_output = cached_single_enc_output.repeat(num_active, 1, 1)
                
                batch_src_len = src_len_b.repeat(num_active)
                batch_trg_list = [bm[0] for bm in active_beams]
                
                # ==========================================
                # 🔥【修正這裡】把它壓扁成 1D (Cat)，不要 Stack 成 2D！
                # ==========================================
                # 因為 PositionalEncoding 是用 seq_len 累加去算位置的，它預期輸入是一條長龍
                batch_trg = torch.cat(batch_trg_list, dim=0)
                    
                # 這裡計算每個 beam 的長度
                batch_trg_len = torch.tensor([seq.size(0) for seq in batch_trg_list], device=device, dtype=torch.int32)
                
                # 呼叫 Decoder forward
                logits = self.forward(
                    src_input_ids=None, # 用不到
                    trg_input_ids=batch_trg,  # 這裡是 1D 了
                    src_seq_len=batch_src_len, 
                    trg_seq_len=batch_trg_len,
                    enc_output=batch_enc_output 
                )
                
                # 取最後一個 token 的 logits
                # 因為我們已經壓扁了，所以要根據 cumsum 找出每個句子的最後一個字
                cu_lens = torch.cumsum(batch_trg_len, dim=0)
                last_indices = cu_lens - 1
                next_logits = logits[last_indices]
                
                log_probs = F.log_softmax(next_logits, dim=-1)
                
                candidates = []
                for i, (seq, score, _) in enumerate(active_beams):
                    # Repetition Penalty
                    if repetition_penalty != 1.0:
                        uniq = torch.unique(seq)
                        current_log_probs = log_probs[i].clone()
                        current_log_probs[uniq] = torch.where(current_log_probs[uniq] < 0, 
                                                            current_log_probs[uniq] * repetition_penalty, 
                                                            current_log_probs[uniq] / repetition_penalty)
                    else:
                        current_log_probs = log_probs[i]
                    
                    if t < min_len:
                        current_log_probs[eos_id] = -float("inf")

                    topk_scores, topk_ids = torch.topk(current_log_probs, beam_size)
                    
                    for s, tid in zip(topk_scores, topk_ids):
                        new_score = score + s.item()
                        new_id = tid.item()
                        
                        new_seq = torch.cat([seq, torch.tensor([new_id], device=device)])
                        
                        if new_id == eos_id:
                            # Length Normalization
                            final_score = new_score / (len(new_seq) ** len_penalty)
                            completed_beams.append((new_seq, final_score, True))
                        else:
                            candidates.append((new_seq, new_score, False))
                
                # 選出最好的 top K
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
                
                if len(completed_beams) >= beam_size:
                    break
                
            if completed_beams:
                best_seq = max(completed_beams, key=lambda x: x[1])[0]
            elif beams:
                best_seq = beams[0][0]
            else:
                best_seq = torch.tensor([], device=device)
                
            toks = best_seq.tolist()
            if eos_id in toks: toks = toks[:toks.index(eos_id)]
            output_text.append(self.tokenizer.decode(toks, skip_special_tokens=True))
            
        return output_text

    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None: return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(self.encoder.embeddings.tok_embeddings.weight)
        self.output_projection.weight = self.decoder.trg_word_emb.weight