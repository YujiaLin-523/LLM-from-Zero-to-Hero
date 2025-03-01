import torch
import torch.nn as nn
import math


# V1：简化版本
class SelfAttention_V1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super(). __init__()
        self.hidden_dim = hidden_dim

        # 定义三个投影层
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_dim], Q, K, V: [batch_size, seq_len, hidden_dim]
        Q = self.query_projection(x)
        K = self.key_projection(x)  
        V = self.value_projection(x)

        # attention_value: [batch_size, seq_len, seq_len]
        attention_value = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

        # attention_weight: [batch_size, seq_len, seq_len]
        attention_weight = nn.functional.softmax(attention_value, dim=-1) 
        
        # output: [batch_size, seq_len, hidden_dim]
        output = torch.matmul(attention_weight, V)
        return output
    

# 测试部分
x = torch.rand(3, 5, 728)
model = SelfAttention_V1()
output = model(x)
print("SelfAttention_V1 output shape: ", output.shape)


# V2：合并Q，K，V矩阵，优化小模型的运行速度
class SelfAttention_V2(nn.Module):
    def __init__(self, dim: int) -> None:
        super(). __init__()
        self.dim = dim

        # 定义一个投影层
        self.projection = nn.Linear(dim, dim * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, dim], Q, K, V: [batch_size, seq_len, dim * 3]
        QKV = self.projection(x)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        # attention_value: [batch_size, seq_len, seq_len]
        attention_value = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)

        # attention_weight: [batch_size, seq_len, seq_len]
        attention_weight = nn.functional.softmax(attention_value, dim=-1) 
        
        # output: [batch_size, seq_len, dim]
        output = torch.matmul(attention_weight, V)
        return output


# 测试部分
x = torch.rand(3, 5, 4)
model = SelfAttention_V2(4)
output = model(x)
print("SelfAttention_V2 output shape: ", output.shape)


# V3：加入Dropout，Mask机制
class SelfAttention_V3(nn.Module):
    def __init__(self, dim: int = 728, dropout_rate: float = 0.1) -> None:
        super(). __init__()
        self.dim = dim
        self.projection = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch_size, seq_len, dim], Q, K, V: [batch_size, seq_len, dim * 3]
        QKV = self.projection(x)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        # attention_weight: [batch_size, seq_len, seq_len]
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask == 0, -1e20)
            attention_weight = nn.functional.softmax(attention_weight, dim=-1)
            attention_weight = self.dropout(attention_weight)

            # output: [batch_size, seq_len, dim]
            output = torch.matmul(attention_weight, V)
            output = self.output_projection(output)

            return output


# 测试部分
x = torch.rand(3, 4, 2)

# mask: [batch_size, seq_len, seq_len]
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
mask = mask.unsqueeze(1).repeat(1, 4, 1)

model = SelfAttention_V3(2)
output = model(x, mask)
print("SelfAttention_V3 output shape: ", output.shape)


# V4：面试写法
class SelfAttention_V4(nn.Module):
    def __init__(self, dim: int = 728, dropout_rate: float = 0.1) -> None:
        super(). __init__()
        self.dim = dim
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch_size, seq_len, dim], Q, K, V: [batch_size, seq_len, dim]
        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # attention_weight: [batch_size, seq_len, seq_len]
        attention_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dim)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask == 0, -1e20)
            attention_weight = nn.functional.softmax(attention_weight, dim=-1)
            attention_weight = self.dropout(attention_weight) # 先dropout再与V矩阵相乘

            # output: [batch_size, seq_len, dim]
            output = (attention_weight @ V)
            output = self.output_projection(output)

            return output


# 测试部分
x = torch.rand(3, 4, 2)
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)

# 增加一个维度，使得mask与x的维度相同
mask = mask.unsqueeze(1).repeat(1, 4, 1) 
model = SelfAttention_V4(2)
output = model(x, mask)
print("SelfAttention_V4 output shape: ", output.shape)
