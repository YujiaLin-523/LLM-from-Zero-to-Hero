import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num # hidden_dim = head_num * head_dim

        # hidden_dim -> head_num * head_dim
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.size()

        # Q, K, V: (batch_size, seq_len, hidden_dim)
        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # Q, K, V: (batch_size, seq_len, hidden_dim) -> (batch_size, head_num, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # attention_weight: (batch_size, head_num, seq_len, seq_len)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask == 0, -1e20)

        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)
        # print("Attention weight shape:", attention_weight.shape)

        output = attention_weight @ V

        # 使用contiguous()函数来将tensor变成在内存中连续分布的形式
        output = output.transpose(1, 2).contiguous() 

        # output: (batch_size, head_num, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        output = output.view(batch_size, seq_len, self.hidden_dim)

        return output


# 测试部分
HEAD_NUM = 8
HIDDEN_DIM = 128

x = torch.randn(3, 4, HIDDEN_DIM)
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)

# mask: (batch_size, seq_len) -> (batch_size, head_num, seq_len, seq_len)
mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, HEAD_NUM, 1, 1)

# mask = mask.unsqueeze(1)
# print("Mask shape:", mask.shape)
# mask = mask.unsqueeze(2)
# print("Mask shape:", mask.shape)
# mask = mask.repeat(1, HEAD_NUM, 1, 1)
# print("Mask shape:", mask.shape)

model = MultiHeadSelfAttention(HIDDEN_DIM, HEAD_NUM)
output = model(x, mask)

print("The shape of input tensor:", x.shape)
print("The shape of output tensor:", output.shape)
