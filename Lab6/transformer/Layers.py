''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers_upgrade import PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash
class DecoderLayer_Flash(nn.Module):
    ''' Compose with three layers using Flash Attention '''
    def __init__(self, d_model, d_inner, n_head, d_qkv, dropout=0.1):
        super(DecoderLayer_Flash, self).__init__()
        self.slf_attn = MultiHeadSelfAttention_Flash(
            n_head, d_model, d_qkv, dropout=dropout, causal=True
        )
        self.enc_attn = MultiHeadCrossAttention_Flash(
            n_head, d_model, d_qkv, dropout=dropout, causal=False
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, dec_seq_lens, enc_output, enc_seq_lens):
        #################YOUR CODE HERE#################
        # 1. Self-Attention
        # 2. Encoder-Decoder Attention
        # 3. Position-wise Feed-Forward Network
        ################################################
        # 1. Self-Attention (輸入是 Decoder 自己)
        # dec_output 的形狀: (總字數, d_model)
        dec_output = self.slf_attn(dec_input, dec_seq_lens)
        
        # 2. Encoder-Decoder Attention (拿剛剛算好的結果，去查 Encoder 的原文)
        # Q 是 dec_output, K/V 是 enc_output
        dec_output = self.enc_attn(
            dec_output,   # Q (Decoder)
            enc_output,   # K, V (Encoder)
            dec_seq_lens, # Q 的長度
            enc_seq_lens  # K, V 的長度
        )
        # 3. Position-wise Feed-Forward Network (最後過一層整理)
        dec_output = self.pos_ffn(dec_output)

        return dec_output