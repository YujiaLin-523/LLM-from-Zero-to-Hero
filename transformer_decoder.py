import torch
import torch.nn as nn
import math


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.dropout_rate = dropout_rate

        # Initialize Multi-Head Attention
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.att_dropout = nn.Dropout(dropout_rate)
        self.att_layernorm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Initialize Feed-Forward Network
        self.upscaling = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.downscaling = nn.Linear(4 * hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_layernorm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def AttentionLayer(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:     
        self.Q = Q
        self.K = K
        self.V = V

        # attention_weights: (batch_size, head_num, seq_len, seq_len)
        attention_weights = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.tril()
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        else:
            mask = attention_weights.ones_like(attention_weights).tril()
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.att_dropout(attention_weights)

        # output: (batch_size, head_num, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        output = attention_weights @ V
        batch_size, _, seq_len, _ = output.shape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_projection(output)

        return output
        

    def MultiHeadAttention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_dim) -> (batch_size, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = x.shape

        # Q, K, V: (batch_size, seq_len, hidden_dim) -> (batch_size, head_num, seq_len, head_dim)
        Q = self.query_projection(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = self.key_projection(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = self.value_projection(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        MHA_output = self.AttentionLayer(Q, K, V, mask)

        return MHA_output

    def FeedForwardNetwork(self, x: torch.Tensor) -> torch.Tensor:
        upscaling = self.upscaling(x)
        upscaling = self.activation(upscaling)
        downscaling = self.downscaling(upscaling)
        downscaling = self.ffn_dropout(downscaling)
        FFN_output = self.ffn_layernorm(x + downscaling)

        return FFN_output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.MultiHeadAttention(x, mask)
        x = self.FeedForwardNetwork(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim=64, head_num=8) for _ in range(5)
            ]
        )

        self.embedding = nn.Embedding(num_embeddings=12, embedding_dim=64)
        self.output_projection = nn.Linear(64, 12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.output_projection(x)

        return x

# 测试部分
HIDDE_DIM = 128
HEAD_NUM = 8

x = torch.randint(0, 12, (3, 4))
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
)

# mask: (batch_size, seq_len) -> (batch_size, head_num, seq_len, seq_len)
mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, HEAD_NUM, 1, 1)

model = Decoder()
output = model(x, mask)
print("Output tensor shape: ", output.shape)
